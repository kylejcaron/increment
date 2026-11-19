# Error codes

Search for the full `CodedError.code`, or choose its family below. Codes and context keys are stable; message wording may change.

Message templates show the cause and any corrective guidance declared by the registry. Braced fields are filled from the error's context at runtime. For context-dependent messages, read the raised message; the linked rule is its source.

Context keys include template fields, explicitly declared keys and named renderer parameters (including optional parameters). Registering modules include the declaration and every module that reuses it.

## Registered refusal codes

[`absorption.*`](#family-absorption) · [`adjust.*`](#family-adjust) · [`allocation.*`](#family-allocation) · [`analysis.*`](#family-analysis) · [`arm.*`](#family-arm) · [`arm_planning.*`](#family-arm_planning) · [`artifact.*`](#family-artifact) · [`artifact_contract.*`](#family-artifact_contract) · [`breakout.*`](#family-breakout) · [`cate.*`](#family-cate) · [`compatibility.*`](#family-compatibility) · [`contrast.*`](#family-contrast) · [`conversion_inference.*`](#family-conversion_inference) · [`dashboard.*`](#family-dashboard) · [`decision.*`](#family-decision) · [`definition.*`](#family-definition) · [`design.*`](#family-design) · [`estimation.*`](#family-estimation) · [`evidence.*`](#family-evidence) · [`facade.*`](#family-facade) · [`family.*`](#family-family) · [`frame.*`](#family-frame) · [`identification.*`](#family-identification) · [`impute.*`](#family-impute) · [`install.*`](#family-install) · [`integrity.*`](#family-integrity) · [`kernel.*`](#family-kernel) · [`logged_policy.*`](#family-logged_policy) · [`model.*`](#family-model) · [`moments.*`](#family-moments) · [`package.*`](#family-package) · [`plan.*`](#family-plan) · [`power.*`](#family-power) · [`query.*`](#family-query) · [`readout.*`](#family-readout) · [`report.*`](#family-report) · [`reporting.*`](#family-reporting) · [`results.*`](#family-results) · [`sensitivity.*`](#family-sensitivity) · [`sequential.*`](#family-sequential) · [`simulate.*`](#family-simulate) · [`source.*`](#family-source) · [`store.*`](#family-store) · [`tables.*`](#family-tables) · [`unit_cycle.*`](#family-unit_cycle) · [`wire.*`](#family-wire)

<span id="family-absorption"></span>

## `absorption.*`

<span id="absorption.absorb_factor_needs"></span>

### `absorption.absorb_factor_needs`

**Cause and message:**

<p>absorb_factor needs exactly two arms; got {arms}. Filter the table to one treated arm before calling.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`

**Registering modules:** `increment.absorption`

<span id="absorption.control_group_present"></span>

### `absorption.control_group_present`

**Cause and message:**

<p>control group {control_group!r} not present; arms are {arms}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `control_group`

**Registering modules:** `increment.absorption`

<span id="absorption.summary_duplicate_rows"></span>

### `absorption.summary_duplicate_rows`

**Cause and message:**

<p>summary has duplicate rows for {named}; absorb_factor needs exactly one row per (factor level, arm); a duplicate would silently overwrite the earlier row&#x27;s moments and bias the effect. Aggregate to one row per cell upstream (GROUP BY {factor!r}, group_id).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `factor`, `named`

**Registering modules:** `increment.absorption`

<span id="absorption.summary_eager_dataframe"></span>

### `absorption.summary_eager_dataframe`

**Cause and message:**

<p>summary must be an eager dataframe (e.g. .to_pyarrow() on an ibis Table)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.absorption`

<span id="absorption.summary_missing_columns"></span>

### `absorption.summary_missing_columns`

**Cause and message:**

<p>summary is missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.absorption`

<span id="family-adjust"></span>

## `adjust.*`

<span id="adjust.variance_reduction.unsupported"></span>

### `adjust.variance_reduction.unsupported`

**Cause and message:**

<p>variance_reduction={variance_reduction!r} is not supported under adjustment {method!r}; the adjustment estimators consume unit frames, not the moments the reduction is defined on; it would be silently ignored rather than applied.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`, `variance_reduction`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="adjust.winsorization.percentile_unsupported"></span>

### `adjust.winsorization.percentile_unsupported`

**Cause and message:**

<p>estimate_ate does not support percentile-winsorized metrics ({metric}); use fixed-value winsorization</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="family-allocation"></span>

## `allocation.*`

<span id="allocation.alpha.underflow"></span>

### `allocation.alpha.underflow`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L257).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `denominator`, `multiplier`, `numerator`

**Registering modules:** `increment.compatibility`

<span id="allocation.invalid_config"></span>

### `allocation.invalid_config`

**Cause and message:**

<p>Invalid allocation configuration: {field} {reason}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `reason`

**Registering modules:** `increment.assignment._config`

<span id="allocation.invalid_target"></span>

### `allocation.invalid_target`

**Cause and message:**

<p>Invalid allocation target: {field} {reason}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `reason`

**Registering modules:** `increment.assignment._config`

<span id="allocation.invalid_unit"></span>

### `allocation.invalid_unit`

**Cause and message:**

<p>Invalid allocation unit: {field} {reason}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `reason`

**Registering modules:** `increment.assignment._config`

<span id="family-analysis"></span>

## `analysis.*`

<span id="analysis.planning_baseline.cluster_icc_not_estimable"></span>

### `analysis.planning_baseline.cluster_icc_not_estimable`

**Cause and message:**

<p>cluster ICC is not estimable from the control arm&#x27;s {control_clusters} cluster(s) over {control_units} units; needs at least 2 clusters, and not exactly one unit per cluster; pass cluster_icc explicitly to Baseline instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `control_clusters`, `control_units`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.covariate_source_unavailable"></span>

### `analysis.planning_baseline.covariate_source_unavailable`

**Cause and message:**

<p>metric {metric!r} declares CUPED but this source&#x27;s moments do not carry the covariate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.cuped_no_residual_variance"></span>

### `analysis.planning_baseline.cuped_no_residual_variance`

**Cause and message:**

<p>metric {metric!r}: its CUPED covariate explains all of the control arm&#x27;s variance, which usually means the covariate repeats the outcome; declare a pre-period covariate, or plan without CUPED</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.design_unavailable"></span>

### `analysis.planning_baseline.design_unavailable`

**Cause and message:**

<p>metric {metric!r}: no declared design on this source; planning_baseline needs a control group</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.quantile_insufficient_data"></span>

### `analysis.planning_baseline.quantile_insufficient_data`

**Cause and message:**

<p>metric {metric!r}: fewer than 2 control-arm values on this source; a quantile planning baseline needs at least 2 observations</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.quantile_source_unavailable"></span>

### `analysis.planning_baseline.quantile_source_unavailable`

**Cause and message:**

<p>metric {metric!r} has no per-unit control-arm values on this source; quantile planning baselines need from_definitions, from_unit_day_artifact, or from_unit_summary</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.trigger_evidence_unavailable"></span>

### `analysis.planning_baseline.trigger_evidence_unavailable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_planning_baseline.py#L99).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `missing`, `trigger`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.trigger_membership_unavailable"></span>

### `analysis.planning_baseline.trigger_membership_unavailable`

**Cause and message:**

<p>metric {metric!r}: clustered triggered planning needs assigned and contributing cluster memberships to derive participation; republish source evidence with assigned and triggered counts</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `analyzed_clusters`, `assigned_clusters`, `control_group`, `metric`, `route`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.trigger_metric_population_unavailable"></span>

### `analysis.planning_baseline.trigger_metric_population_unavailable`

**Cause and message:**

<p>metric {metric!r}: this source supplies outcomes for {observed_units} of {eligible_units} trigger-eligible control units; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `control_group`, `eligible_units`, `metric`, `observed_units`, `route`, `trigger`

**Registering modules:** `increment._planning_baseline`

<span id="analysis.planning_baseline.trigger_population_inconsistent"></span>

### `analysis.planning_baseline.trigger_population_inconsistent`

**Cause and message:**

<p>metric {metric!r}: assigned/analyzed cluster memberships imply analyzed mean size {derived_mean}, but analyzed rows report {observed_mean}; retain both memberships from the same pilot</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `analyzed_clusters`, `assigned_clusters`, `assigned_units`, `derived_mean`, `metric`, `observed_mean`, `route`, `trigger_rate`

**Registering modules:** `increment._planning_baseline`

<span id="family-arm"></span>

## `arm.*`

<span id="arm.adjustment.cluster_cuped"></span>

### `arm.adjustment.cluster_cuped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.adjustment.cluster_prior"></span>

### `arm.adjustment.cluster_prior`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.adjustment.sequential_cuped"></span>

### `arm.adjustment.sequential_cuped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.adjustment.sequential_prior"></span>

### `arm.adjustment.sequential_prior`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.absorption_undeclared"></span>

### `arm.baseline.absorption_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.cluster_size_required"></span>

### `arm.baseline.cluster_size_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.compliance_undeclared"></span>

### `arm.baseline.compliance_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.cuped_undeclared"></span>

### `arm.baseline.cuped_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.iid_cluster_knobs"></span>

### `arm.baseline.iid_cluster_knobs`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.baseline.triggering_undeclared"></span>

### `arm.baseline.triggering_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.inference.cluster"></span>

### `arm.inference.cluster`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.metric.quantile_cluster"></span>

### `arm.metric.quantile_cluster`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.metric.quantile_cuped"></span>

### `arm.metric.quantile_cuped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.metric.quantile_sequential"></span>

### `arm.metric.quantile_sequential`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="arm.metric.unsupported"></span>

### `arm.metric.unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="family-arm_planning"></span>

## `arm_planning.*`

<span id="arm_planning.guardrail_neutral_requires_direction"></span>

### `arm_planning.guardrail_neutral_requires_direction`

**Cause and message:**

<p>preferred_direction=&#x27;neutral&#x27; declares no direction, so a guardrail still can&#x27;t be one-sided; pass preferred_direction=&#x27;increase&#x27;/&#x27;decrease&#x27; or an explicit one-sided alternative=&#x27;greater&#x27;/&#x27;less&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternative`, `preferred_direction`

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.guardrail_preferred_direction_conflict"></span>

### `arm_planning.guardrail_preferred_direction_conflict`

**Cause and message:**

<p>alternative disagrees with the declared preferred_direction</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.guardrail_requires_one_sided"></span>

### `arm_planning.guardrail_requires_one_sided`

**Cause and message:**

<p>a guardrail is evaluated at full alpha, one-sided; pass preferred_direction= or an explicit one-sided alternative=</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.inference_spec_adjustments_unplannable"></span>

### `arm_planning.inference_spec_adjustments_unplannable`

**Cause and message:**

<p>predeclared CUPED adjustments are a runtime-only declaration; planning admits CUPED by passing a Baseline with cuped_rho set to the solver call</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.inference_spec_expected_n_unplannable"></span>

### `arm_planning.inference_spec_expected_n_unplannable`

**Cause and message:**

<p>expected_decision_sample_size is solved for, not declared; read it from PowerResult.inference_to_declare after sizing</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.inference_spec_registration_unplannable"></span>

### `arm_planning.inference_spec_registration_unplannable`

**Cause and message:**

<p>planning derives its own registration-free boundary; pass InferenceSpec(kind=&#x27;asymptotic_mean&#x27;) without registration=</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.inference_spec_segments_unplannable"></span>

### `arm_planning.inference_spec_segments_unplannable`

**Cause and message:**

<p>standard() fixes view=&#x27;total&#x27; and segmented=False, so it cannot represent declared segments {rejected_segments!r} without dropping them; use the full ArmPlanningProcedure constructor with view=&#x27;breakout&#x27; and segmented=True for an explicitly segmented fixed-horizon plan; standard() provides no segmented asymptotic planning construction (route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `rejected_segments`, `route`, `segmented`, `view`

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.preferred_direction_without_guardrail_role"></span>

### `arm_planning.preferred_direction_without_guardrail_role`

**Cause and message:**

<p>preferred_direction= is only meaningful with role=&#x27;guardrail&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.secondaries_without_secondary_role"></span>

### `arm_planning.secondaries_without_secondary_role`

**Cause and message:**

<p>secondaries= is only meaningful with role=&#x27;secondary&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.secondary_requires_count"></span>

### `arm_planning.secondary_requires_count`

**Cause and message:**

<p>role=&#x27;secondary&#x27; requires secondaries=&lt;count&gt;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="arm_planning.sequential_secondary_one_arm"></span>

### `arm_planning.sequential_secondary_one_arm`

**Cause and message:**

<p>a sequential secondary&#x27;s runtime registration supports exactly one treatment arm; pass comparisons=1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="family-artifact"></span>

## `artifact.*`

<span id="artifact.artifact_contract.refusal_code"></span>

### `artifact.artifact_contract.refusal_code`

**Cause and message:**

<p>artifact refusal requires code</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.context.legacy_exclusion_acknowledged"></span>

### `artifact.context.legacy_exclusion_acknowledged`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.context.mismatch"></span>

### `artifact.context.mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.coverage_dates_date"></span>

### `artifact.coverage_dates_date`

**Cause and message:**

<p>coverage dates must be date values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.cuped.scalar_breakout"></span>

### `artifact.cuped.scalar_breakout`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.digest.json"></span>

### `artifact.digest.json`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.nonfinite"></span>

### `artifact.digest.nonfinite`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.nullable"></span>

### `artifact.digest.nullable`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.primary_key"></span>

### `artifact.digest.primary_key`

**Cause and message:**

<p>artifact digest primary key {primary_key!r} (field {field!r}): {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `primary_key`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.range"></span>

### `artifact.digest.range`

**Cause and message:**

<p>artifact digest {quantity} (field {field!r}) must lie in [{minimum}, {maximum}]; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `field`, `maximum`, `minimum`, `quantity`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.recanonicalize"></span>

### `artifact.digest.recanonicalize`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.role"></span>

### `artifact.digest.role`

**Cause and message:**

<p>artifact digest relation role: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.row"></span>

### `artifact.digest.row`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.rows"></span>

### `artifact.digest.rows`

**Cause and message:**

<p>artifact digest rows: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.schema"></span>

### `artifact.digest.schema`

**Cause and message:**

<p>artifact digest schema declaration {position} (field {field!r}): {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `position`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.timestamp"></span>

### `artifact.digest.timestamp`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.digest.type"></span>

### `artifact.digest.type`

**Cause and message:**

<p>artifact digest field {field!r}: {constraint}; got {value!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactDigestError`

**Context keys:** `constraint`, `field`, `route`, `value`

**Registering modules:** `increment.query.artifact_digest`

<span id="artifact.evidence.unavailable"></span>

### `artifact.evidence.unavailable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.extension.invalid"></span>

### `artifact.extension.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.extension.missing"></span>

### `artifact.extension.missing`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.first_ds_last"></span>

### `artifact.first_ds_last`

**Cause and message:**

<p>first_ds must be &lt;= last_ds</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.format.unsupported"></span>

### `artifact.format.unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.generation.dropped"></span>

### `artifact.generation.dropped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.identifier.unsafe"></span>

### `artifact.identifier.unsafe`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.manifest.invalid"></span>

### `artifact.manifest.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.manifest.unreferenced_relation"></span>

### `artifact.manifest.unreferenced_relation`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.metric.binding_mismatch"></span>

### `artifact.metric.binding_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.on_mixed_assignment"></span>

### `artifact.on_mixed_assignment`

**Cause and message:**

<p>on_mixed_assignment must be &#x27;error&#x27;, &#x27;warn&#x27;, or &#x27;exclude&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.operation.unsupported"></span>

### `artifact.operation.unsupported`

**Cause and message:**

<p>artifact source cannot perform {operation!r} for {request!r}; available capability is {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `offered`, `operation`, `request`, `route`

**Registering modules:** `increment.query.artifact_reader`

<span id="artifact.refresh.context_mismatch"></span>

### `artifact.refresh.context_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.refresh.invalid_ref"></span>

### `artifact.refresh.invalid_ref`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.relation.digest_mismatch"></span>

### `artifact.relation.digest_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.relation.invalid"></span>

### `artifact.relation.invalid`

**Cause and message:**

<p>{relation}: {check}; {detail}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `check`, `detail`, `relation`

**Registering modules:** `increment.query.artifact_reader`

<span id="artifact.relation.schema_mismatch"></span>

### `artifact.relation.schema_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.request.trigger_metric_scope"></span>

### `artifact.request.trigger_metric_scope`

**Cause and message:**

<p>trigger_measure_stats must name exactly one metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.artifact`

<span id="artifact.snapshot.mixed"></span>

### `artifact.snapshot.mixed`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="artifact.store.namespace"></span>

### `artifact.store.namespace`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/session.py#L119).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`

**Registering modules:** `increment.query.session`

<span id="family-artifact_contract"></span>

## `artifact_contract.*`

<span id="artifact_contract.encouragement_uptake_conflict"></span>

### `artifact_contract.encouragement_uptake_conflict`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/artifact_contract.py#L330).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `ArtifactContractError`

**Context keys:** `message`

**Registering modules:** `increment.query.artifact_contract`, `increment.query.artifact_extensions`, `increment.query.artifact_publish`, `increment.query.artifact_reader`, `increment.query.source`

<span id="family-breakout"></span>

## `breakout.*`

<span id="breakout.breakout.exactly_one_lift"></span>

### `breakout.breakout.exactly_one_lift`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/breakout/estimates.py#L186).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.breakout.point_unavailable"></span>

### `breakout.breakout.point_unavailable`

**Cause and message:**

<p>a finite point estimate is unavailable for this breakout row</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.coerce_type_datetime"></span>

### `breakout.coerce_type_datetime`

**Cause and message:**

<p>cannot coerce {v!r} (type {type_name}) to datetime.date</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `type_name`, `v`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.compliance_late_view"></span>

### `breakout.compliance_late_view`

**Cause and message:**

<p>compliance and late require view=&#x27;asof&#x27; with an encouragement design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.daily_group_summary"></span>

### `breakout.daily_group_summary`

**Cause and message:**

<p>daily_group_summary missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.daily_lift.exactly_one_unavailable"></span>

### `breakout.daily_lift.exactly_one_unavailable`

**Cause and message:**

<p>exactly one of lift or unavailable must be set</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.daily_lift.point_unavailable"></span>

### `breakout.daily_lift.point_unavailable`

**Cause and message:**

<p>a finite point estimate is unavailable for this daily lift row</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.daily_metric.exactly_one_value"></span>

### `breakout.daily_metric.exactly_one_value`

**Cause and message:**

<p>exactly one of value or unavailable must be set</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.metric_found_group"></span>

### `breakout.metric_found_group`

**Cause and message:**

<p>Metric(s) {unknown_metrics} found in group_summary but not in the metrics list. Declared metrics: {declared_names}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared_names`, `unknown_metrics`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.price_rollout_metrics"></span>

### `breakout.price_rollout_metrics`

**Cause and message:**

<p>cannot price a rollout for metrics whose preferred_direction is not &#x27;increase&#x27;: {misdirected}; the selection rule keeps segments whose lift EXCEEDS the cost, which presumes larger is better, so pricing a decrease-preferred metric would recommend the segments that moved worst. Drop it from metrics=, and from the estimates too if they carry it: pricing follows the estimates, while the declaration this refusal reads comes from metrics=, so removing it from only one of the two either leaves it priced at 0.0 or re-raises this same error</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `misdirected`

**Registering modules:** `increment.breakout.rollout`

<span id="breakout.quantile"></span>

### `breakout.quantile`

**Cause and message:**

<p>quantile metric(s) {names!r} have no moments representation, so {method} cannot serve them; {reason}. {remedy}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `method`, `names`, `reason`, `remedy`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.retention.completion"></span>

### `breakout.retention.completion`

**Cause and message:**

<p>{fn_name}: completed_windows_only=True is contradictory for retention metric(s) {names!r}; their bands are open on the right, so no unit&#x27;s window ever completes and the gate would admit nothing. Drop completed_windows_only to get the cumulative monitoring series, or declare threshold_days: [a, b] to bound the band.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fn_name`, `names`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="breakout.retention.daily"></span>

### `breakout.retention.daily`

**Cause and message:**

<p>{fn_name}: RetentionMetric(s) {names!r} have no valid independent-per-day snapshot; &#x27;returned within the observation band&#x27; is not a property of a single day. Use the as-of view (run_asof/run_asof_lift, calendar axis) or the cohort view (run_daily/run_daily_lift on a retention metric, indexed by exposure date), or pass metrics= to exclude it.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fn_name`, `names`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.retention.encouragement"></span>

### `breakout.retention.encouragement`

**Cause and message:**

<p>{fn_name}: retention metric(s) {names!r} are not supported under an encouragement design; its maturity gate changes the first-stage population by outcome metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `fn_name`, `names`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.retention.unbounded"></span>

### `breakout.retention.unbounded`

**Cause and message:**

<p>{fn_name}: retention metric(s) {names!r} declare an unbounded band, so their outcome never completes; on this view every value would be a lower bound on itself, biased hardest for the least-observed units. {remedy}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fn_name`, `names`, `remedy`, `supported_view`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="breakout.rollout_cost_names"></span>

### `breakout.rollout_cost_names`

**Cause and message:**

<p>rollout_cost= names metrics these estimates do not carry: {unknown} (known: {known}); refusing rather than silently pricing a rollout at the wrong threshold</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `known`, `unknown`

**Registering modules:** `increment.breakout.rollout`

<span id="breakout.rollout_cost_rollout"></span>

### `breakout.rollout_cost_rollout`

**Cause and message:**

<p>rollout_cost={{{name!r}: {value}}}: a rollout cost is a relative lift and must be finite and &gt; -1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `value`

**Registering modules:** `increment.breakout.rollout`

<span id="breakout.run_breakout_correction"></span>

### `breakout.run_breakout_correction`

**Cause and message:**

<p>run_breakout: correction must be &#x27;none&#x27;, &#x27;bonferroni&#x27;, or &#x27;bh&#x27;, got {correction!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_breakout_dimension"></span>

### `breakout.run_breakout_dimension`

**Cause and message:**

<p>run_breakout: dimension column {dimension!r} not found in summary. Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `dimension`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_breakout_duplicate"></span>

### `breakout.run_breakout_duplicate`

**Cause and message:**

<p>run_breakout: duplicate summary row for segment {dimension}={value!r}, metric={metric!r}, group_id={group_id!r}; each (segment, metric, arm) cell&#x27;s moments must arrive as one pre-aggregated row; duplicates would silently discard one row&#x27;s units (duplicate control: last row wins the control lookup) or double-count the segment downstream (duplicate treatment: two output rows for one cell inflate segment_heterogeneity&#x27;s k).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`, `group_id`, `metric`, `value`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_breakout_group"></span>

### `breakout.run_breakout_group`

**Cause and message:**

<p>run_breakout: group_summary missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_breakout_methods"></span>

### `breakout.run_breakout_methods`

**Cause and message:**

<p>run_breakout: methods=[] requests zero estimation methods; every declared (metric, arm) pair would then be reported as unestimable with a misleading &#x27;no data&#x27; warning, when the real reason is that no method was asked for. Pass methods=None for the default ([Method(name=&#x27;unadjusted&#x27;)]) or a non-empty list of Methods.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_daily_lift"></span>

### `breakout.run_daily_lift`

**Cause and message:**

<p>run_daily_lift: correction must be &#x27;none&#x27; or &#x27;bonferroni&#x27;, got {correction!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_daily_lift_dimension_found"></span>

### `breakout.run_daily_lift_dimension_found`

**Cause and message:**

<p>run_daily_lift: dimension column {dimension!r} not found in summary. Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `dimension`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_daily_lift_ds_column_found"></span>

### `breakout.run_daily_lift_ds_column_found`

**Cause and message:**

<p>run_daily_lift: &#x27;ds&#x27; column not found in summary. Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_daily_lift_metric_declared"></span>

### `breakout.run_daily_lift_metric_declared`

**Cause and message:**

<p>Metric(s) {unknown} found in group_summary but not in the metrics list. Declared metrics: {declared}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `unknown`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.run_daily_lift_plan_missing_procedures"></span>

### `breakout.run_daily_lift_plan_missing_procedures`

**Cause and message:**

<p>run_daily_lift: compiled plan is missing procedure(s) for {missing_procedures!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing_procedures`

**Registering modules:** `increment.breakout.estimates`

<span id="breakout.segment.lift_none_excluded"></span>

### `breakout.segment.lift_none_excluded`

**Cause and message:**

<p>lift=None requires excluded to explain the unavailable row</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment.point_unavailable"></span>

### `breakout.segment.point_unavailable`

**Cause and message:**

<p>a finite point estimate is unavailable for this excluded segment row</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_absolute_interval"></span>

### `breakout.segment_contrast_absolute_interval`

**Cause and message:**

<p>segment_contrast requires an interval on each segment&#x27;s lift (lb/ub/level all set) to recover its variance</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_dimension"></span>

### `breakout.segment_contrast_dimension`

**Cause and message:**

<p>segment_contrast: dimension_value={dimension_value_a!r} and {dimension_value_b!r} do not share the same (metric, method, group_id, dimension, source, estimand, value_scale) grouping ({key_a!r} vs {key_b!r}). Scope `estimates` to a single grouping before calling segment_contrast.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension_value_a`, `dimension_value_b`, `key_a`, `key_b`

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_estimate"></span>

### `breakout.segment_contrast_estimate`

**Cause and message:**

<p>segment_contrast requires an estimate on both segments; unavailable rows cannot be contrasted</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_expected"></span>

### `breakout.segment_contrast_expected`

**Cause and message:**

<p>segment_contrast: expected exactly 1 row for dimension_value={label!r}, got {row_count}. Scope `estimates` to a single (metric, method, group_id, dimension, source, estimand, value_scale) combination before calling segment_contrast.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `label`, `row_count`

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_open_interval_variance"></span>

### `breakout.segment_contrast_open_interval_variance`

**Cause and message:**

<p>segment_contrast could not recover a positive, finite standard error from the open interval&#x27;s calibrated endpoint</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_contrast_reference_fixed_horizon"></span>

### `breakout.segment_contrast_reference_fixed_horizon`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/breakout/heterogeneity.py#L115).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension_value`, `dof`, `inference`, `reference_df`, `reference_kind`

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.segment_heterogeneity_rebuild"></span>

### `breakout.segment_heterogeneity_rebuild`

**Cause and message:**

<p>segment_heterogeneity: cannot rebuild {metric!r}/{dimension_value!r}&#x27;s relative-scale interval at alpha={alpha}; the row is alternative={alternative!r}, dof={dof!r}, inference={inference!r}, reference_kind={reference_kind!r}, reference_df={reference_df!r}, and a rebuild from its stored statistics would be wrong or unresolved: a symmetric two-sided Normal interval would silently misrepresent its shape or reference (including a t sampling reference from clustering or Welch), and an exact binomial inversion at this alpha whose endpoint search did not reach its resolution is not reported. Pass an alpha matching this row&#x27;s own level, or exclude it before correcting.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `alternative`, `dimension_value`, `dof`, `inference`, `metric`, `reference_df`, `reference_kind`

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.tau_prior_scale"></span>

### `breakout.tau_prior_scale`

**Cause and message:**

<p>tau_prior_scale must be &gt; 0, got {tau_prior_scale}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tau_prior_scale`

**Registering modules:** `increment.breakout.heterogeneity`

<span id="breakout.to_frame_infer"></span>

### `breakout.to_frame_infer`

**Cause and message:**

<p>to_frame(): cannot infer the schema from an empty sequence; pass model=&lt;the result model class&gt; explicitly, or call &lt;Model&gt;Estimates(estimates).to_frame() instead, which always knows its own model even when empty.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.projection`

<span id="breakout.to_frame_model"></span>

### `breakout.to_frame_model`

**Cause and message:**

<p>to_frame(): model={model} does not match estimates[0]&#x27;s actual type {inferred}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `inferred`, `model`

**Registering modules:** `increment.breakout.projection`

<span id="breakout.unknown_estimand_supported"></span>

### `breakout.unknown_estimand_supported`

**Cause and message:**

<p>unknown estimand(s) {unknown_estimands}; supported: {supported}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `supported`, `unknown_estimands`

**Registering modules:** `increment.breakout.estimates`

<span id="family-cate"></span>

## `cate.*`

<span id="cate.cate_supported_quantile"></span>

### `cate.cate_supported_quantile`

**Cause and message:**

<p>CATE is not supported for quantile metric {metric!r}; the per-unit outcome is the unit&#x27;s total, so the fitted effect would be a conditional MEAN effect reported under a quantile metric&#x27;s name. Declare the column as a mean metric instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.cate`

<span id="cate.cluster.covariate_conflict"></span>

### `cate.cluster.covariate_conflict`

**Cause and message:**

<p>{caller}: {column!r} is this source&#x27;s declared cluster identity (source.context.cluster) and cannot also be requested as an interact/adjust covariate or observational adjustment column, even though it may look numeric; cluster identity is metadata, never a model feature.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `column`

**Registering modules:** `increment.cate`

<span id="cate.cluster.identity_unavailable"></span>

### `cate.cluster.identity_unavailable`

**Cause and message:**

<p>{caller}: source.context.cluster == {column!r} declares a cluster identity, but unit_frame served no aligned cluster_id column; a nonconformant MomentSource implementation. Refusing rather than silently treating this population as unclustered.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `column`

**Registering modules:** `increment.cate`

<span id="cate.cluster.intervention_grain_unavailable"></span>

### `cate.cluster.intervention_grain_unavailable`

**Cause and message:**

<p>{caller}: source.context.intervention_grain == &#x27;cluster&#x27; declares a whole-cluster policy deployment, but this source resolved no cluster identity (source.context.cluster is None); there is no cluster grain to intervene on.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`

**Registering modules:** `increment.cate`

<span id="cate.cluster.randomization_arm_purity"></span>

### `cate.cluster.randomization_arm_purity`

**Cause and message:**

<p>{caller}: cluster {cluster!r} contains both treatment arms ({values}) under a declared cluster-randomized source (source.context.design.mechanism == &#x27;randomized&#x27; with a declared cluster); randomization at the cluster grain requires every cluster to be a pure arm. An observational design (declaring the dependence honestly) preserves mixed-treatment clusters instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `cluster`, `values`

**Registering modules:** `increment.cate`

<span id="cate.contrasts_exactly_two"></span>

### `cate.contrasts_exactly_two`

**Cause and message:**

<p>{caller} contrasts exactly two arms, but {metric!r} has {n_arms}: {arms}. Against control {control!r} the treatment indicator would pool {others} into a single arm, which estimates a different quantity. Restrict the source to two arms and call once per contrast</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `caller`, `control`, `metric`, `n_arms`, `others`

**Registering modules:** `increment.cate`

<span id="cate.control_arm_present"></span>

### `cate.control_arm_present`

**Cause and message:**

<p>control arm {control!r} is not present in {metric!r}; arms are {arms}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `control`, `metric`

**Registering modules:** `increment.cate`

<span id="cate.covariate.is_outcome"></span>

### `cate.covariate.is_outcome`

**Cause and message:**

<p>{caller}: covariate {covariate!r} is the outcome of metric {metric!r}. Covariates must be measured before exposure; remove the outcome from interact, adjust and the design&#x27;s adjustment set.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `covariate`, `metric`

**Registering modules:** `increment.cate`

<span id="cate.does_support_ratio"></span>

### `cate.does_support_ratio`

**Cause and message:**

<p>{caller} does not support ratio metric {metric!r}: conditional ratio effects require modeling denominator variation. Use readouts.run() for the arm-level ratio effect, or model numerator and denominator separately.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `caller`, `metric`

**Registering modules:** `increment.cate`

<span id="cate.identification.randomized_only"></span>

### `cate.identification.randomized_only`

**Cause and message:**

<p>{caller} requires a randomized MomentSource (source.context.design.mechanism == &#x27;randomized&#x27;), got {mechanism!r}; estimate_cate&#x27;s per-unit predictions do not have a doubly robust observational implementation</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `mechanism`

**Registering modules:** `increment.cate`

<span id="cate.identification.unsupported_max_smd"></span>

### `cate.identification.unsupported_max_smd`

**Cause and message:**

<p>{caller} does not implement the observational max_smd={max_smd!r} balance gate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `max_smd`

**Registering modules:** `increment.cate`

<span id="cate.identification.unsupported_mechanism"></span>

### `cate.identification.unsupported_mechanism`

**Cause and message:**

<p>{caller} does not support identification mechanism {mechanism!r}; supported mechanisms are {supported}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `mechanism`, `supported`

**Registering modules:** `increment.cate`

<span id="cate.identification.unsupported_missing_policy"></span>

### `cate.identification.unsupported_missing_policy`

**Cause and message:**

<p>{caller} does not implement observational adjustment missing={missing!r}; this CATE path currently requires missing=&#x27;refuse&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `missing`

**Registering modules:** `increment.cate`

<span id="cate.metric_declared_source"></span>

### `cate.metric_declared_source`

**Cause and message:**

<p>metric {metric!r} is not declared on this source; available: {available}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `metric`

**Registering modules:** `increment.cate`

<span id="family-compatibility"></span>

## `compatibility.*`

<span id="compatibility.denominator"></span>

### `compatibility.denominator`

**Cause and message:**

<p>denominator must be &gt;= 1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.compatibility`

<span id="compatibility.multiplier"></span>

### `compatibility.multiplier`

**Cause and message:**

<p>multiplier must be &gt;= 1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.compatibility`

<span id="compatibility.support_unsupported"></span>

### `compatibility.support_unsupported`

**Cause and message:**

<p>support must be Unsupported, got {support_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `support_type`

**Registering modules:** `increment.compatibility`

<span id="family-contrast"></span>

## `contrast.*`

<span id="contrast.decision"></span>

### `contrast.decision`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="contrast.inference"></span>

### `contrast.inference`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="contrast.metric"></span>

### `contrast.metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/compatibility.py#L240).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.compatibility`, `increment.estimation.adjust`, `increment.estimation.arm_contract`

<span id="family-conversion_inference"></span>

## `conversion_inference.*`

<span id="conversion_inference.finite_sample.cuped"></span>

### `conversion_inference.finite_sample.cuped`

**Cause and message:**

<p>method {method!r} is CUPED-adjusted (through {adjusted_by}) and requests conversion_inference=&#x27;finite_sample&#x27;: CUPED adjusts the outcome by a fitted covariate slope, so the contrast is no longer a pair of raw binomial counts and has no finite-sample test inversion. Use conversion_inference=&#x27;auto&#x27; (CUPED then adjusts on the asymptotic route) or drop the CUPED adjustment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `adjusted_by`, `method`

**Registering modules:** `increment._finite_sample_refusals`

<span id="conversion_inference.finite_sample.metric_type"></span>

### `conversion_inference.finite_sample.metric_type`

**Cause and message:**

<p>conversion_inference=&#x27;finite_sample&#x27; applies only to conversion and retention metrics, and this metric has type {metric_type!r}; drop it or use conversion_inference=&#x27;auto&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment._finite_sample_refusals`

<span id="family-dashboard"></span>

## `dashboard.*`

<span id="dashboard.invalid_config"></span>

### `dashboard.invalid_config`

**Cause and message:**

<p>dashboard configuration is unusable: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.dashboard._data`

<span id="dashboard.invalid_snapshot"></span>

### `dashboard.invalid_snapshot`

**Cause and message:**

<p>dashboard snapshot is inconsistent: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.dashboard._data`

<span id="dashboard.invalid_theme"></span>

### `dashboard.invalid_theme`

**Cause and message:**

<p>invalid dashboard theme {field}: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `reason`

**Registering modules:** `increment.dashboard._theme`

<span id="dashboard.invalid_view"></span>

### `dashboard.invalid_view`

**Cause and message:**

<p>this dashboard view is unavailable: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.dashboard._data`

<span id="dashboard.population_unavailable"></span>

### `dashboard.population_unavailable`

**Cause and message:**

<p>dashboard population {population!r} was not captured</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `population`

**Registering modules:** `increment.dashboard._data`

<span id="dashboard.unsupported_experiment"></span>

### `dashboard.unsupported_experiment`

**Cause and message:**

<p>the dashboard supports ordinary two-arm randomized experiments: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.dashboard._data`

<span id="family-decision"></span>

## `decision.*`

<span id="decision.absolute_arm.procedures_fixed_inference"></span>

### `decision.absolute_arm.procedures_fixed_inference`

**Cause and message:**

<p>absolute procedures require fixed inference</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.decision`

<span id="decision.arm_decision.one_sided_alpha"></span>

### `decision.arm_decision.one_sided_alpha`

**Cause and message:**

<p>one-sided alpha must be &lt; 0.5 for alpha-doubled display level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.decision`, `increment.estimation.decision_types`

<span id="decision.arm_planning.scalar_power_unsupported_family"></span>

### `decision.arm_planning.scalar_power_unsupported_family`

**Cause and message:**

<p>scalar power does not support BH/e-BH family planning</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.arm_planning.uncorrected_family_size"></span>

### `decision.arm_planning.uncorrected_family_size`

**Cause and message:**

<p>uncorrected planning requires family_size=1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.compiled_decision.procedure_mapping_key"></span>

### `decision.compiled_decision.procedure_mapping_key`

**Cause and message:**

<p>procedure mapping key {key!r} does not match metric {metric!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `key`, `metric`

**Registering modules:** `increment.decision`

<span id="decision.contrast_context.duplicate_metrics"></span>

### `decision.contrast_context.duplicate_metrics`

**Cause and message:**

<p>contrast metric names must be unique: duplicates {metrics}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metrics`

**Registering modules:** `increment.decision`

<span id="decision.contrast_context.procedure_metrics"></span>

### `decision.contrast_context.procedure_metrics`

**Cause and message:**

<p>contrast procedures must match declared metrics: missing {missing}, undeclared {undeclared}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`, `undeclared`

**Registering modules:** `increment.decision`

<span id="decision.contrast_context.reference_mismatch"></span>

### `decision.contrast_context.reference_mismatch`

**Cause and message:**

<p>contrast references must match procedure references for metrics {metrics}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metrics`

**Registering modules:** `increment.decision`

<span id="decision.contrast_decision.reference_metric"></span>

### `decision.contrast_decision.reference_metric`

**Cause and message:**

<p>procedure and prospective reference must name the same metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.decision_types`

<span id="decision.family.axes_contain_duplicates"></span>

### `decision.family.axes_contain_duplicates`

**Cause and message:**

<p>family axes must not contain duplicates: {axes!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `axes`

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.family.axes_contain_unknown"></span>

### `decision.family.axes_contain_unknown`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/arm_contract.py#L61).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unknown`

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.family.axes_use_canonical"></span>

### `decision.family.axes_use_canonical`

**Cause and message:**

<p>family axes must use canonical order {allowed!r}, got {axes!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allowed`, `axes`

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.family.nofamily_membership_mark"></span>

### `decision.family.nofamily_membership_mark`

**Cause and message:**

<p>NoFamily membership cannot mark member=True</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.decision`

<span id="decision.family.policy_least_one"></span>

### `decision.family.policy_least_one`

**Cause and message:**

<p>{kind} family policy requires at least one axis</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kind`

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.family.uncorrected_policy_empty"></span>

### `decision.family.uncorrected_policy_empty`

**Cause and message:**

<p>uncorrected family policy requires empty axes</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="decision.multiplicity.bh"></span>

### `decision.multiplicity.bh`

**Cause and message:**

<p>q is only valid for BH multiplicity, got {correction!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`

**Registering modules:** `increment._multiplicity`

<span id="decision.multiplicity.guarantee_mismatch"></span>

### `decision.multiplicity.guarantee_mismatch`

**Cause and message:**

<p>multiplicity guarantee {guarantee!r} does not match correction {correction!r}; the exact matrix is none-&gt;none, bonferroni-&gt;fwer, bh/e_bh-&gt;fdr</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`, `guarantee`

**Registering modules:** `increment._multiplicity`

<span id="decision.multiplicity.validate_policy"></span>

### `decision.multiplicity.validate_policy`

**Cause and message:**

<p>{correction} multiplicity requires q</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`

**Registering modules:** `increment._multiplicity`

<span id="decision.p_value_evidence.finite_unit_interval"></span>

### `decision.p_value_evidence.finite_unit_interval`

**Cause and message:**

<p>p_value must be finite and in [0, 1], got {p_value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `p_value`

**Registering modules:** `increment.estimation.decision_types`

<span id="decision.relative_arm.null_lift_greater"></span>

### `decision.relative_arm.null_lift_greater`

**Cause and message:**

<p>relative null_lift must be greater than -1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.decision`

<span id="decision.relative_decision.one_sided_nominal"></span>

### `decision.relative_decision.one_sided_nominal`

**Cause and message:**

<p>one-sided nominal alpha must be &lt; 0.5</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.arm_contract`

<span id="family-definition"></span>

## `definition.*`

<span id="definition.active.metric_window_days"></span>

### `definition.active.metric_window_days`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.adjustment_set.adjustmentset_covariates_contains"></span>

### `definition.adjustment_set.adjustmentset_covariates_contains`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/design.py#L80).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dups`

**Registering modules:** `increment.semantics.design`

<span id="definition.already_canonical_json"></span>

### `definition.already_canonical_json`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.analysis.appears_both_metric"></span>

### `definition.analysis.appears_both_metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.analysis.appears_more_once"></span>

### `definition.analysis.appears_more_once`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact.context_format_disagrees"></span>

### `definition.artifact.context_format_disagrees`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact.context_sha256_does"></span>

### `definition.artifact.context_sha256_does`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact.trigger_metric_scope"></span>

### `definition.artifact.trigger_metric_scope`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact.trigger_timestamp_timezone"></span>

### `definition.artifact.trigger_timestamp_timezone`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_datetimes_timezone"></span>

### `definition.artifact_datetimes_timezone`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_extension.definition_sha256_does"></span>

### `definition.artifact_extension.definition_sha256_does`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_extension.source_provenance_sha256"></span>

### `definition.artifact_extension.source_provenance_sha256`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_extension.unsupported_version"></span>

### `definition.artifact_extension.unsupported_version`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_json_values"></span>

### `definition.artifact_json_values`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_relation.primary_key_contain"></span>

### `definition.artifact_relation.primary_key_contain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_relation.primary_key_members"></span>

### `definition.artifact_relation.primary_key_members`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.artifact_relation.row_count_integer"></span>

### `definition.artifact_relation.row_count_integer`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.assignment.carryover_order_range"></span>

### `definition.assignment.carryover_order_range`

**Cause and message:**

<p>carryover_order={carryover_order} must satisfy 0 &lt;= carryover_order &lt; observation_steps={observation_steps}; an order at or past observation_steps would discard the entire retained window</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `carryover_order`, `observation_steps`

**Registering modules:** `increment.semantics.assignment`

<span id="definition.assignment.carryover_order_type"></span>

### `definition.assignment.carryover_order_type`

**Cause and message:**

<p>carryover_order must be a non-boolean int</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.assignment`

<span id="definition.assignment.reject_bool"></span>

### `definition.assignment.reject_bool`

**Cause and message:**

<p>expected an int, got a bool</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.assignment`, `increment.semantics.design`

<span id="definition.assignment_populations_contain"></span>

### `definition.assignment_populations_contain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.assignment_populations_include"></span>

### `definition.assignment_populations_include`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.exposures_primary_key"></span>

### `definition.base_relations.exposures_primary_key`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.exposures_relation_use"></span>

### `definition.base_relations.exposures_relation_use`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.measure_stats_primary"></span>

### `definition.base_relations.measure_stats_primary`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.measure_stats_relation"></span>

### `definition.base_relations.measure_stats_relation`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.share_artifact_id"></span>

### `definition.base_relations.share_artifact_id`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.base_relations.share_generation_id"></span>

### `definition.base_relations.share_generation_id`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.breakout.ambiguous_source"></span>

### `definition.breakout.ambiguous_source`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.check_observational.covariate_could"></span>

### `definition.check_observational.covariate_could`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.check_observational.covariate_property_found"></span>

### `definition.check_observational.covariate_property_found`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.check_observational.covariate_source_found"></span>

### `definition.check_observational.covariate_source_found`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.check_observational.covariate_unit_but"></span>

### `definition.check_observational.covariate_unit_but`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.contain_control_characters"></span>

### `definition.contain_control_characters`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.cuped.ambiguous_covariate_source"></span>

### `definition.cuped.ambiguous_covariate_source`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.cuped_preperiod.window_bounds_integers"></span>

### `definition.cuped_preperiod.window_bounds_integers`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.cuped_preperiod.window_satisfy_window"></span>

### `definition.cuped_preperiod.window_satisfy_window`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.day_boundary_accepts"></span>

### `definition.day_boundary_accepts`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.duplicate_property_name"></span>

### `definition.dim.duplicate_property_name`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.source_declares_no"></span>

### `definition.dim.source_declares_no`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.source_no_validity"></span>

### `definition.dim.source_no_validity`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.source_versioned_validity"></span>

### `definition.dim.source_versioned_validity`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.validity_mutually_exclusive_encoding"></span>

### `definition.dim.validity_mutually_exclusive_encoding`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.validity_ranges_need"></span>

### `definition.dim.validity_ranges_need`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.validity_requires_one_encoding"></span>

### `definition.dim.validity_requires_one_encoding`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.dim.validity_valid_from"></span>

### `definition.dim.validity_valid_from`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.duplicate_scalar"></span>

### `definition.duplicate_scalar`

**Cause and message:**

<p>scalar key {field!r} is declared in more than one YAML file {files!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `field`, `files`, `path`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="definition.duplicates"></span>

### `definition.duplicates`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.encode_json_object"></span>

### `definition.encode_json_object`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.encouragement_uptake.window_days_integer"></span>

### `definition.encouragement_uptake.window_days_integer`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.expected_datetime_bool"></span>

### `definition.expected_datetime_bool`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.expected_iso_8601_epoch"></span>

### `definition.expected_iso_8601_epoch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.expected_iso_8601_numeric_string"></span>

### `definition.expected_iso_8601_numeric_string`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.binding_margin_margin"></span>

### `definition.experiment.binding_margin_margin`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.cluster_combined_n"></span>

### `definition.experiment.cluster_combined_n`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.cluster_names_same"></span>

### `definition.experiment.cluster_names_same`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_binding_exactly_one"></span>

### `definition.experiment.covariate_binding_exactly_one`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_measure_needs_preperiod"></span>

### `definition.experiment.covariate_measure_needs_preperiod`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_measure_numeric"></span>

### `definition.experiment.covariate_measure_numeric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_name_reserved"></span>

### `definition.experiment.covariate_name_reserved`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_names_unique"></span>

### `definition.experiment.covariate_names_unique`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.covariate_source_requires_property"></span>

### `definition.experiment.covariate_source_requires_property`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.design_tuning_key_not_yaml"></span>

### `definition.experiment.design_tuning_key_not_yaml`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.end_before_start"></span>

### `definition.experiment.end_before_start`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.intervention_grain_without_cluster"></span>

### `definition.experiment.intervention_grain_without_cluster`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.metric_cuped_linear_needs_covariates"></span>

### `definition.experiment.metric_cuped_linear_needs_covariates`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.metric_declares_cuped"></span>

### `definition.experiment.metric_declares_cuped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.observation_end_before_end"></span>

### `definition.experiment.observation_end_before_end`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.observation_end_requires_end"></span>

### `definition.experiment.observation_end_requires_end`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.experiment.timezone_aware_but"></span>

### `definition.experiment.timezone_aware_but`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.exposure.filters_apply_fact"></span>

### `definition.exposure.filters_apply_fact`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.exposure.needs_exactly_one"></span>

### `definition.exposure.needs_exactly_one`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.exposure.sql_set_but"></span>

### `definition.exposure.sql_set_but`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.fact.source_declares_propert"></span>

### `definition.fact.source_declares_propert`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.fact.source_points_propert"></span>

### `definition.fact.source_points_propert`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.fact.source_renames_its"></span>

### `definition.fact.source_renames_its`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.fact.source_renames_propert"></span>

### `definition.fact.source_renames_propert`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.fact_source.missing_columns"></span>

### `definition.fact_source.missing_columns`

**Cause and message:**

<p>fact source &#x27;{fact_source}&#x27; SQL is missing required column alias(es) {missing!r}; add those aliases to the source SELECT list</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `fact_source`, `missing`

**Registering modules:** `increment.query.fact_resolution`

<span id="definition.fact_source.sql_unresolved"></span>

### `definition.fact_source.sql_unresolved`

**Cause and message:**

<p>fact source &#x27;{fact_source}&#x27; SQL could not be resolved by the connection; check that every relation it reads exists and this connection can read it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `fact_source`

**Registering modules:** `increment.query.fact_resolution`

<span id="definition.filter.op_between_comparable"></span>

### `definition.filter.op_between_comparable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.filter.op_between_values"></span>

### `definition.filter.op_between_values`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.filter.op_exactly_value"></span>

### `definition.filter.op_exactly_value`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.filter.op_exactly_values"></span>

### `definition.filter.op_exactly_values`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.filter.op_least_value"></span>

### `definition.filter.op_least_value`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.filter.values_finite"></span>

### `definition.filter.values_finite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.inference.always_valid_metric_type"></span>

### `definition.inference.always_valid_metric_type`

**Cause and message:**

<p>metric {metric!r} has type {metric_type!r}; InferenceSpec(kind=&#x27;always_valid&#x27;) without a registration monitors conversion and retention metrics with the exact Bernoulli e-process. Monitor this metric with InferenceSpec(kind=&#x27;asymptotic_mean&#x27;), or declare an explicit SequentialRegistration</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `metric`, `metric_type`

**Registering modules:** `increment.sequential_source`

<span id="definition.inference.baseline_rate_domain"></span>

### `definition.inference.baseline_rate_domain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.inference.baseline_rate_route"></span>

### `definition.inference.baseline_rate_route`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.invalid"></span>

### `definition.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.json"></span>

### `definition.json`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.lowercase_64_hex"></span>

### `definition.lowercase_64_hex`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.method.covariates"></span>

### `definition.method.covariates`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.method.methodspec_name_cuped"></span>

### `definition.method.methodspec_name_cuped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_base.margin_margin_abs"></span>

### `definition.metric_base.margin_margin_abs`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_base.rollout_cost_preferred"></span>

### `definition.metric_base.rollout_cost_preferred`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_both_relative"></span>

### `definition.metric_both_relative`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_margin_abs_requires_direction"></span>

### `definition.metric_margin_abs_requires_direction`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_margin_abs_requires_non_neutral_direction"></span>

### `definition.metric_margin_abs_requires_non_neutral_direction`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_margin_implies_unrepresentable_null_lift"></span>

### `definition.metric_margin_implies_unrepresentable_null_lift`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_margin_non"></span>

### `definition.metric_margin_non`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_margin_requires_direction"></span>

### `definition.metric_margin_requires_direction`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.metric_unknown_alternative"></span>

### `definition.metric_unknown_alternative`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.migration.guardrails"></span>

### `definition.migration.guardrails`

**Cause and message:**

<p>experiment &#x27;{experiment}&#x27;: &#x27;guardrails&#x27; moved to plan.guardrails; declare it as `plan: {{guardrails: [...]}}` instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `experiment`, `path`

**Registering modules:** `increment.semantics.loader`

<span id="definition.migration.metrics"></span>

### `definition.migration.metrics`

**Cause and message:**

<p>experiment &#x27;{experiment}&#x27;: &#x27;metrics&#x27; moved to plan.secondaries; declare it as `plan: {{secondaries: [...]}}` instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `experiment`, `path`

**Registering modules:** `increment.semantics.loader`

<span id="definition.models.reject_bool"></span>

### `definition.models.reject_bool`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.multiplicity.bh_q_only_valid_for_bh"></span>

### `definition.multiplicity.bh_q_only_valid_for_bh`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.multiplicity.bh_requires_q"></span>

### `definition.multiplicity.bh_requires_q`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.non_empty_string"></span>

### `definition.non_empty_string`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.observational.ambiguous_covariate_source"></span>

### `definition.observational.ambiguous_covariate_source`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.path.missing"></span>

### `definition.path.missing`

**Cause and message:**

<p>path does not exist: {path}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `path`

**Registering modules:** `increment.semantics.loader`

<span id="definition.quantile.metric_use_avg"></span>

### `definition.quantile.metric_use_avg`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.ratio.metric_numerator_denominator"></span>

### `definition.ratio.metric_numerator_denominator`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.ratio.metric_use_avg"></span>

### `definition.ratio.metric_use_avg`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.ratio_metric.numerator_denominator_differ"></span>

### `definition.ratio_metric.numerator_denominator_differ`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.relation_bind_manifest"></span>

### `definition.relation_bind_manifest`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.relation_primary_key"></span>

### `definition.relation_primary_key`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.relation_role"></span>

### `definition.relation_role`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.resolve_breakout.experiment_property_could"></span>

### `definition.resolve_breakout.experiment_property_could`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.resolve_breakout.experiment_property_found"></span>

### `definition.resolve_breakout.experiment_property_found`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.resolve_breakout.experiment_property_references"></span>

### `definition.resolve_breakout.experiment_property_references`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.resolve_breakout.experiment_unit_but"></span>

### `definition.resolve_breakout.experiment_unit_but`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.retention.metric_window_days"></span>

### `definition.retention.metric_window_days`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.retention.threshold_days_lower_bound_non_negative"></span>

### `definition.retention.threshold_days_lower_bound_non_negative`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.retention.threshold_days_non_negative"></span>

### `definition.retention.threshold_days_non_negative`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.retention.threshold_days_upper_exceeds_lower"></span>

### `definition.retention.threshold_days_upper_exceeds_lower`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.root"></span>

### `definition.root`

**Cause and message:**

<p>YAML root in {path} must be a {expected}, got {value_type}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `expected`, `path`, `route`, `value_type`

**Registering modules:** `increment.semantics.loader`

<span id="definition.section"></span>

### `definition.section`

**Cause and message:**

<p>definition field {field} in {path} must be a {expected}, got {value_type}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `expected`, `field`, `path`, `route`, `value_type`

**Registering modules:** `increment.semantics.loader`

<span id="definition.shared.catalog"></span>

### `definition.shared.catalog`

**Cause and message:**

<p>Shared metric document {path}:{line}:{column}: {reason}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `catalog_code`, `column`, `line`, `path`, `reason`, `route`

**Registering modules:** `increment.semantics.shared`

<span id="definition.shared.include"></span>

### `definition.shared.include`

**Cause and message:**

<p>Shared metric declaration in {path}: {reason}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `path`, `reason`, `route`

**Registering modules:** `increment.semantics.shared`

<span id="definition.shared.metric_unsupported"></span>

### `definition.shared.metric_unsupported`

**Cause and message:**

<p>Shared metric {metric!r} cannot be used in {target}: {reason}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `metric`, `reason`, `route`, `target`

**Registering modules:** `increment.semantics.shared`

<span id="definition.single_sql_identifier"></span>

### `definition.single_sql_identifier`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.site_volume.first_ds_last"></span>

### `definition.site_volume.first_ds_last`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.site_volume.measure_keys_empty"></span>

### `definition.site_volume.measure_keys_empty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.site_volume.measure_keys_unique"></span>

### `definition.site_volume.measure_keys_unique`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.site_volume.metric_names_empty"></span>

### `definition.site_volume.metric_names_empty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.site_volume.metric_names_unique"></span>

### `definition.site_volume.metric_names_unique`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.source.sql_empty"></span>

### `definition.source.sql_empty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.sql"></span>

### `definition.sql`

**Cause and message:**

<p>{label}: SQL admission failed with {error_type}: {error}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `error`, `error_type`, `label`, `path`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="definition.sql.dialect"></span>

### `definition.sql.dialect`

**Cause and message:**

<p>{label}: unknown SQL dialect {dialect!r}; admission cannot be verified against a parser that does not exist; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dialect`, `label`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="definition.sql.not_read_only"></span>

### `definition.sql.not_read_only`

**Cause and message:**

<p>{label} must contain exactly one read-only query; found {violation} {construct!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `construct`, `dialect`, `label`, `route`, `violation`

**Registering modules:** `increment.semantics.loader`

<span id="definition.sql.parse"></span>

### `definition.sql.parse`

**Cause and message:**

<p>invalid SQL in {label}: {error}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `dialect`, `error`, `label`, `line`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="definition.sql.statement_count"></span>

### `definition.sql.statement_count`

**Cause and message:**

<p>{label} must contain exactly one read-only query; parsed {statements} statement(s), {empty} of them empty; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dialect`, `empty`, `label`, `route`, `statements`

**Registering modules:** `increment.semantics.loader`

<span id="definition.surrounding_whitespace"></span>

### `definition.surrounding_whitespace`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.total.metric_use_avg"></span>

### `definition.total.metric_use_avg`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.total.metric_window_days"></span>

### `definition.total.metric_window_days`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_cycle.aggregation"></span>

### `definition.unit_cycle.aggregation`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_cycle.assignment"></span>

### `definition.unit_cycle.assignment`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_cycle.cycles"></span>

### `definition.unit_cycle.cycles`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_cycle.groups"></span>

### `definition.unit_cycle.groups`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_cycle.nonempty"></span>

### `definition.unit_cycle.nonempty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_cycle.references"></span>

### `definition.unit_cycle.references`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/unit_cycle.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.unit_cycle`

<span id="definition.unit_day.created_at_timezone"></span>

### `definition.unit_day.created_at_timezone`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_extension_absent"></span>

### `definition.unit_day.manifest_extension_absent`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_extensions_contain"></span>

### `definition.unit_day.manifest_extensions_contain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_extensions_use"></span>

### `definition.unit_day.manifest_extensions_use`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_first_ds"></span>

### `definition.unit_day.manifest_first_ds`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_measure_keys"></span>

### `definition.unit_day.manifest_measure_keys`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_measures_use"></span>

### `definition.unit_day.manifest_measures_use`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_metric_bindings"></span>

### `definition.unit_day.manifest_metric_bindings`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_metric_bindings_canonical_order"></span>

### `definition.unit_day.manifest_metric_bindings_canonical_order`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_metric_names"></span>

### `definition.unit_day.manifest_metric_names`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.manifest_sha256_does"></span>

### `definition.unit_day.manifest_sha256_does`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.metric_references_undeclared"></span>

### `definition.unit_day.metric_references_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.ratio_metric_references"></span>

### `definition.unit_day.ratio_metric_references`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.site_volume_keys"></span>

### `definition.unit_day.site_volume_keys`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.site_volume_keys_absent_from_manifest"></span>

### `definition.unit_day.site_volume_keys_absent_from_manifest`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.site_volume_request"></span>

### `definition.unit_day.site_volume_request`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.trigger_measure_anchor"></span>

### `definition.unit_day.trigger_measure_anchor`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.trigger_measure_keys"></span>

### `definition.unit_day.trigger_measure_keys`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.unit_day.trigger_measure_metric"></span>

### `definition.unit_day.trigger_measure_metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.utf"></span>

### `definition.utf`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.validation"></span>

### `definition.validation`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/loader.py#L131).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `errors`, `path`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="definition.winsorization.finite"></span>

### `definition.winsorization.finite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.winsorization.least_one_bound"></span>

### `definition.winsorization.least_one_bound`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.winsorization.lower_percentile_below"></span>

### `definition.winsorization.lower_percentile_below`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.winsorization.lower_value_below"></span>

### `definition.winsorization.lower_value_below`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.winsorization.set_percentile_fixed"></span>

### `definition.winsorization.set_percentile_fixed`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/models.py#L77).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `message`

**Registering modules:** `increment.semantics.loader`, `increment.semantics.models`

<span id="definition.yaml"></span>

### `definition.yaml`

**Cause and message:**

<p>YAML parse error in {path}: {error}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `DefinitionError`

**Context keys:** `column`, `error`, `line`, `path`, `route`

**Registering modules:** `increment.semantics.loader`

<span id="family-design"></span>

## `design.*`

<span id="design.allocation.invalid"></span>

### `design.allocation.invalid`

**Cause and message:**

<p>allocation {allocation!r} is not a valid assignment split for control group {control_group!r}: it must be non-empty, contain {control_group!r} as a key, and every weight must be finite and strictly positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allocation`, `control_group`

**Registering modules:** `increment.semantics.design`

<span id="design.allocation_scheme.incompatible"></span>

### `design.allocation_scheme.incompatible`

**Cause and message:**

<p>allocation_scheme {allocation_scheme!r} is incompatible with {design!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allocation_scheme`, `design`

**Registering modules:** `increment.semantics.design`, `increment.semantics.models`

<span id="design.mechanism_unsupported"></span>

### `design.mechanism_unsupported`

**Cause and message:**

<p>design mechanism {mechanism!r} is unsupported; supported mechanisms are {supported}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mechanism`, `supported`

**Registering modules:** `increment.semantics.design`

<span id="family-estimation"></span>

## `estimation.*`

<span id="estimation.absorption.absorption_interval_half"></span>

### `estimation.absorption.absorption_interval_half`

**Cause and message:**

<p>absorption interval half-width is not representable in float64 (critical value {crit:.6g} x se {se:.6g}); the effect is not estimable at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `crit`, `se`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.all_moment_arrays"></span>

### `estimation.absorption.all_moment_arrays`

**Cause and message:**

<p>all moment arrays must have the same length, got sizes {lengths}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lengths`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.contain_exact_integer"></span>

### `estimation.absorption.contain_exact_integer`

**Cause and message:**

<p>{name} must contain exact integer unit counts, got {arr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arr`, `name`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.contains_non_finite"></span>

### `estimation.absorption.contains_non_finite`

**Cause and message:**

<p>{name} contains non-finite values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.need_least_levels"></span>

### `estimation.absorption.need_least_levels`

**Cause and message:**

<p>need at least {min_levels} levels with &gt;= {min_cell_n} units in both arms; got {n_usable}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `min_cell_n`, `min_levels`, `n_usable`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.non_negative"></span>

### `estimation.absorption.non_negative`

**Cause and message:**

<p>{name} must be non-negative, got {arr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arr`, `name`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.one_dimensional_shape"></span>

### `estimation.absorption.one_dimensional_shape`

**Cause and message:**

<p>{name} must be one-dimensional, got shape {arr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arr`, `name`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.absorption.pooling_one"></span>

### `estimation.absorption.pooling_one`

**Cause and message:**

<p>pooling must be one of {pooling_options}, got {pooling!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `pooling`, `pooling_options`

**Registering modules:** `increment.estimation.absorption`

<span id="estimation.adjust.estimate_ate_every"></span>

### `estimation.adjust.estimate_ate_every`

**Cause and message:**

<p>estimate_ate: every declared metric was refused by the requested adjustment method(s); nothing to estimate. See the accompanying UserWarnings for the per-metric refusal reasons.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.adjust`, `increment.plan`, `increment.readouts._common`

<span id="estimation.adjust.method_name_unadjusted"></span>

### `estimation.adjust.method_name_unadjusted`

**Cause and message:**

<p>Method(name=&#x27;unadjusted&#x27;) cannot honor value_scale=&#x27;absolute&#x27;: the unadjusted moments path already reports the absolute pair (abs_diff/abs_se) alongside relative lift on every row, and its rows are confounded; an absolute-native unadjusted row would dress a non-causal difference in the channel built to rescue identified adjusted estimates. Drop &#x27;unadjusted&#x27; from methods or remove the metric from value_scale.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.names_metrics_source"></span>

### `estimation.adjust.names_metrics_source`

**Cause and message:**

<p>{label}= names metrics this source does not declare: {unknown} (declared: {by_name}); refusing rather than silently dropping the request</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `by_name`, `label`, `unknown`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.prior.method_scale"></span>

### `estimation.adjust.prior.method_scale`

**Cause and message:**

<p>prior= cannot span unadjusted log-RR and adjusted linear-relative methods in one request; run separate calls with priors declared on each method&#x27;s parameterization</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.prior.type"></span>

### `estimation.adjust.prior.type`

**Cause and message:**

<p>This prior needs a randomized relative-lift analysis; this contrast does not retain the required relative likelihood. Use an eligible randomized analysis, or an unwrapped Normal prior for this contrast.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.inference`, `increment.plan`

<span id="estimation.adjust.prior_interpreted_call"></span>

### `estimation.adjust.prior_interpreted_call`

**Cause and message:**

<p>prior= cannot be interpreted for a call that reports some metrics on the relative scale and others on the absolute scale: the same Normal would be read as unitless relative lift on one row and metric-units additive lift on another. Drop prior= or make the call scale-uniform.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric"></span>

### `estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric`

**Cause and message:**

<p>margins_abs/null_abs cannot target metric {metric!r}: its rows are absolute-native (value_scale=&#x27;absolute&#x27;), so the abs_diff/abs_se fields the absolute-margin decision reads are None by design. A shifted null for an absolute-native row is a null_lift in the metric&#x27;s own units, which this row does not support. Testing against 0 with alternative= runs but drops the margin, so it answers a different question.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric"></span>

### `estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric`

**Cause and message:**

<p>null_lifts/margins cannot target metric {metric!r} with a nonzero value: its rows are absolute-native (value_scale=&#x27;absolute&#x27;), so `lift` is in the metric&#x27;s own units and a unitless relative null would be compared against an additive interval. A nonzero shifted null is not available on an absolute-native row. Testing against 0 with alternative= runs but drops the margin, so it answers a different question.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.value_scale_names"></span>

### `estimation.adjust.value_scale_names`

**Cause and message:**

<p>value_scale names metric {name!r}: no observational adjustment exists for {metric_type} metrics; the absolute channel changes how an adjusted estimate is REPORTED; it does not extend estimator coverage.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_type`, `name`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust.value_scale_relative"></span>

### `estimation.adjust.value_scale_relative`

**Cause and message:**

<p>value_scale={{{name!r}: {requested!r}}}: must be &#x27;relative&#x27; or &#x27;absolute&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `requested`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="estimation.adjust_common.control_group_found"></span>

### `estimation.adjust_common.control_group_found`

**Cause and message:**

<p>control_group {control_group!r} not found in {metric!r} groups. Available groups: {available}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `control_group`, `metric`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.covariate_dtype"></span>

### `estimation.adjust_common.covariate_dtype`

**Cause and message:**

<p>{method}: adjustment covariate {covariate!r} has dtype {dtype} for metric {metric!r} ({contrast}); an adjustment covariate is numeric (int/float/bool) or categorical (string); a date, time, nested or mixed-type column carries no adjustment meaning. Derive a numeric or categorical pre-exposure column upstream.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `contrast`, `covariate`, `dtype`, `method`, `metric`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.fold_training_split"></span>

### `estimation.adjust_common.fold_training_split`

**Cause and message:**

<p>{method} fold {j} training split has only one arm for metric {metric!r} ({contrast}); cannot fit the propensity/outcome model(s); lower `folds` or supply more data.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `contrast`, `j`, `method`, `metric`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.fold_would_train"></span>

### `estimation.adjust_common.fold_would_train`

**Cause and message:**

<p>{method} fold {j} would train on zero units for metric {metric!r} ({contrast}); the unit-id hash collapsed every unit into one fold; lower `folds`.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `contrast`, `j`, `method`, `metric`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.needs_least_folds"></span>

### `estimation.adjust_common.needs_least_folds`

**Cause and message:**

<p>{method} needs at least 2*folds={needed} units to cross-fit for metric {metric!r} ({contrast}); got {n}. Lower `folds` or supply more data.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `contrast`, `method`, `metric`, `n`, `needed`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.statistically_indistinguishable_from"></span>

### `estimation.adjust_common.statistically_indistinguishable_from`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_adjust/common.py#L163).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `denominator`, `estimator`, `label`, `metric`, `se`, `treatment`

**Registering modules:** `increment.estimation._adjust.common`

<span id="estimation.adjust_common.supported_ratio_metric"></span>

### `estimation.adjust_common.supported_ratio_metric`

**Cause and message:**

<p>{method} is not supported for ratio metric {metric!r}; {method} reweights or residualizes a single per-unit outcome, and a ratio&#x27;s two components would need the weights applied jointly with their covariance retained, which is not implemented for this method. Ratio metrics ARE supported under randomized CUPED (Method.cuped(covariates=...)), which adjusts the numerator and the denominator with a slope each and keeps their induced covariance. Otherwise adjust the numerator and denominator as separate mean metrics.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `correction`, `family`, `method`, `metric`, `role`

**Registering modules:** `increment.estimation._adjust.common`, `increment.estimation.adjust`

<span id="estimation.adjust_dml.dml_residualized_treatment"></span>

### `estimation.adjust_dml.dml_residualized_treatment`

**Cause and message:**

<p>DML residualized treatment has zero variation for metric {metric_name!r} (contrast {contrast!r}); the propensity model perfectly predicts treatment assignment on this data; theta is not identified</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `contrast`, `metric_name`

**Registering modules:** `increment.estimation._adjust.dml`

<span id="estimation.adjust_encoding.predict_called"></span>

### `estimation.adjust_encoding.predict_called`

**Cause and message:**

<p>EncodedLearner.predict called before fit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._adjust.encoding`

<span id="estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge"></span>

### `estimation.adjust_learners.logistic_propensity.logisticpropensity_failed_converge`

**Cause and message:**

<p>LogisticPropensity failed to converge to a finite solution: {res}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `res`

**Registering modules:** `increment.estimation._adjust.learners`

<span id="estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called"></span>

### `estimation.adjust_learners.logistic_propensity.logisticpropensity_predict_called`

**Cause and message:**

<p>LogisticPropensity.predict called before fit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._adjust.learners`

<span id="estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called"></span>

### `estimation.adjust_learners.ridge_outcome.ridgeoutcome_predict_called`

**Cause and message:**

<p>RidgeOutcome.predict called before fit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._adjust.learners`

<span id="estimation.adjust_overlap.arm_cluster_but"></span>

### `estimation.adjust_overlap.arm_cluster_but`

**Cause and message:**

<p>arm {arm!r} has only {clusters} cluster(s), but cross-fitting with {k} folds needs at least {k} per arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `clusters`, `k`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.arm_stratified_cross"></span>

### `estimation.adjust_overlap.arm_stratified_cross`

**Cause and message:**

<p>arm-stratified cross-fitting needs at least two arm labels, got {unique_arms}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unique_arms`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.arm_unit_but"></span>

### `estimation.adjust_overlap.arm_unit_but`

**Cause and message:**

<p>arm {arm!r} has only {indices} unit(s), but cross-fitting with {k} folds needs at least {k} per arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `indices`, `k`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.arms_shape"></span>

### `estimation.adjust_overlap.arms_shape`

**Cause and message:**

<p>arms must be 1-d, got shape {arms}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.arms_unit_ids"></span>

### `estimation.adjust_overlap.arms_unit_ids`

**Cause and message:**

<p>arms and unit_ids must have the same length, got {arms} and {unit_ids}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `unit_ids`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.cluster_arm_needs_two"></span>

### `estimation.adjust_overlap.cluster_arm_needs_two`

**Cause and message:**

<p>treatment-pure cluster contrast needs at least 2 clusters in EACH arm to estimate a between-cluster variance for that arm, got {k_t} treated cluster(s) and {k_c} control cluster(s); a single cluster carries no between-cluster variance contribution to estimate its own arm&#x27;s component from.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k_c`, `k_t`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.cluster_ids_one"></span>

### `estimation.adjust_overlap.cluster_ids_one`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_adjust/overlap.py#L66).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.cluster_needs_two"></span>

### `estimation.adjust_overlap.cluster_needs_two`

**Cause and message:**

<p>cluster covariance needs at least two independent clusters, got {k}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.cross_fitting_needs"></span>

### `estimation.adjust_overlap.cross_fitting_needs`

**Cause and message:**

<p>cross-fitting needs at least 2 folds, got k={k}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.adjust_overlap.fold_ids_needs"></span>

### `estimation.adjust_overlap.fold_ids_needs`

**Cause and message:**

<p>_fold_ids needs string-typed unit ids for a stable hash, got dtype {unit_ids!r}; cast to str before calling (e.g. unit_frame&#x27;s own unit_id column, which is already String-typed)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unit_ids`

**Registering modules:** `increment.estimation._adjust.overlap`

<span id="estimation.armstats.arm_stats.cden3_complete_denominator"></span>

### `estimation.armstats.arm_stats.cden3_complete_denominator`

**Cause and message:**

<p>cden3 requires the complete denominator prerequisite family</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.centered_sum_squares"></span>

### `estimation.armstats.arm_stats.centered_sum_squares`

**Cause and message:**

<p>{what}: centered sum of squares is {c2:.6g}, but it is a sum of squares and cannot be negative; these moments are corrupt or were not produced by a moments builder.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `c2`, `what`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_family_some"></span>

### `estimation.armstats.arm_stats.combine_family_some`

**Cause and message:**

<p>combine() got the {family} family on some partitions and not others ({probe} mixed None/non-None); partitions of one population must agree on which families exist.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `family`, `probe`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_inconsistent_winsorization"></span>

### `estimation.armstats.arm_stats.combine_inconsistent_winsorization`

**Cause and message:**

<p>combine() got inconsistent winsorization metadata in {field}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_needs_least"></span>

### `estimation.armstats.arm_stats.combine_needs_least`

**Cause and message:**

<p>combine() needs at least one partition</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_needs_one"></span>

### `estimation.armstats.arm_stats.combine_needs_one`

**Cause and message:**

<p>combine() needs one study and one metric; got {head_study_id!r}/{head_metric!r} and {a_study_id!r}/{a_metric!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `a_metric`, `a_study_id`, `head_metric`, `head_study_id`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_partitions_declaring"></span>

### `estimation.armstats.arm_stats.combine_partitions_declaring`

**Cause and message:**

<p>combine() got partitions declaring different x-family roles ({roles}); partitions of one population must agree on what the x family carries (covariate / cluster size / uptake total).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `roles`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_partitions_from"></span>

### `estimation.armstats.arm_stats.combine_partitions_from`

**Cause and message:**

<p>combine() got partitions from different arms ({labels}); pass group_id= to label the combined record explicitly.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `labels`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.combine_uptake_family"></span>

### `estimation.armstats.arm_stats.combine_uptake_family`

**Cause and message:**

<p>combine() got the uptake family on some partitions and not others (sum_d mixed None/non-None); partitions of one population must agree on which families exist.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd"></span>

### `estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd`

**Cause and message:**

<p>needs uptake: sum_d/cyd is None (uptake fact not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.cov_yden_needs_cyden"></span>

### `estimation.armstats.arm_stats.cov_yden_needs_cyden`

**Cause and message:**

<p>ratio-family moments require cyden/cden1 (not None)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.cross_field_finite"></span>

### `estimation.armstats.arm_stats.cross_field_finite`

**Cause and message:**

<p>{name} must be finite, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `value`

**Registering modules:** `increment.estimation.armstats`, `increment.estimation.sitewide`

<span id="estimation.armstats.arm_stats.cxd_complete_uptake"></span>

### `estimation.armstats.arm_stats.cxd_complete_uptake`

**Cause and message:**

<p>cxd requires complete x and uptake prerequisite families</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.cxden_complete_denominator"></span>

### `estimation.armstats.arm_stats.cxden_complete_denominator`

**Cause and message:**

<p>cxden requires complete x and denominator prerequisite families</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.group_summary_moment"></span>

### `estimation.armstats.arm_stats.group_summary_moment`

**Cause and message:**

<p>group_summary moment {name!r} is non-finite ({value!r}) for metric={metric!r} group={group_id!r}; {name} must be finite. Rescale the values (for example, change their units) so their centered moments are representable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `name`, `value`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.least_compute_metric"></span>

### `estimation.armstats.arm_stats.least_compute_metric`

**Cause and message:**

<p>n must be at least 2 to compute {quantity} (metric {metric!r}, arm {group_id!r} has n={n}); drop or merge arm {group_id!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `n`, `quantity`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.mean_d_needs_uptake_sum"></span>

### `estimation.armstats.arm_stats.mean_d_needs_uptake_sum`

**Cause and message:**

<p>needs uptake: sum_d is None (uptake fact not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.mean_den_needs_ref_den"></span>

### `estimation.armstats.arm_stats.mean_den_needs_ref_den`

**Cause and message:**

<p>ratio-family moments require ref_den/cden1 (not None)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_covariate_cx2"></span>

### `estimation.armstats.arm_stats.needs_covariate_cx2`

**Cause and message:**

<p>needs covariate: cx2/cx1 is None (CUPED not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_covariate_cxy"></span>

### `estimation.armstats.arm_stats.needs_covariate_cxy`

**Cause and message:**

<p>needs covariate: cxy/cx1 is None (CUPED not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_covariate_ref"></span>

### `estimation.armstats.arm_stats.needs_covariate_ref`

**Cause and message:**

<p>needs covariate: ref_x/cx1 is None (CUPED not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_covariate_uptake"></span>

### `estimation.armstats.arm_stats.needs_covariate_uptake`

**Cause and message:**

<p>needs covariate x uptake: cxd is None. Both a CUPED covariate and an uptake fact must be materialised on the same arm: from a definitions path, set n_pre_periods &gt; 0 on the experiment; from from_unit_summary, declare the column with Method.cuped(covariates=...) alongside uptake=&#x27;&lt;column&gt;&#x27;.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_denominator_cross"></span>

### `estimation.armstats.arm_stats.needs_denominator_cross`

**Cause and message:**

<p>needs x-denominator cross moment: cxden is None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.needs_uptake_cyd"></span>

### `estimation.armstats.arm_stats.needs_uptake_cyd`

**Cause and message:**

<p>needs uptake: cyd/cy2d is None (uptake fact not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.obsolete_armstats_fields"></span>

### `estimation.armstats.arm_stats.obsolete_armstats_fields`

**Cause and message:**

<p>obsolete ArmStats fields are not supported: {obsolete}; weighted estimands must use ScoreStats</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `obsolete`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.partial_family_missing"></span>

### `estimation.armstats.arm_stats.partial_family_missing`

**Cause and message:**

<p>partial {label} family: has {present} but is missing {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `label`, `missing`, `present`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.skew_den_inconsistent"></span>

### `estimation.armstats.arm_stats.skew_den_inconsistent`

**Cause and message:**

<p>denominator skewness for metric={metric!r} group={group_id!r} is unavailable: the carried third moment cden3={cden3!r} is non-finite or exceeds the largest skewness any sample of n={n} values attains; it overflowed or was mis-aggregated upstream; the second moments remain usable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cden3`, `group_id`, `metric`, `n`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.skew_den_needs_cden3"></span>

### `estimation.armstats.arm_stats.skew_den_needs_cden3`

**Cause and message:**

<p>denominator skewness requires cden3/cden2/cden1 (not None); this row&#x27;s producer did not carry the denominator&#x27;s third moment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.skew_den_underflow"></span>

### `estimation.armstats.arm_stats.skew_den_underflow`

**Cause and message:**

<p>denominator skewness for metric={metric!r} group={group_id!r} is unavailable: cden3 is zero while the centered second moment ({cden2!r}) is so small that cubed residuals underflow; rescale the denominator to check skew</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cden2`, `group_id`, `metric`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.sum_d_outside"></span>

### `estimation.armstats.arm_stats.sum_d_outside`

**Cause and message:**

<p>sum_d={sum_d!r} is outside [0, n={n}] for metric={metric!r} group={group_id!r}; an uptake count cannot be negative or exceed the arm&#x27;s own unit count</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `n`, `sum_d`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero"></span>

### `estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero`

**Cause and message:**

<p>sum_d=0 for metric={metric!r} group={group_id!r} but {nonzero} nonzero; no unit took up, so every uptake-masked moment must be exactly zero; these moments were not produced from the same mask</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `nonzero`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.var_den_needs_cden2"></span>

### `estimation.armstats.arm_stats.var_den_needs_cden2`

**Cause and message:**

<p>ratio-family moments require cden2/cden1 (not None)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_cap_counts"></span>

### `estimation.armstats.arm_stats.winsorization_cap_counts`

**Cause and message:**

<p>winsorization cap counts cannot exceed winsor_n</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_lower_percentile"></span>

### `estimation.armstats.arm_stats.winsorization_lower_percentile`

**Cause and message:**

<p>winsorization lower percentile requires a resolved bound</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_metadata_finite"></span>

### `estimation.armstats.arm_stats.winsorization_metadata_finite`

**Cause and message:**

<p>winsorization metadata {name} must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_metadata_winsor"></span>

### `estimation.armstats.arm_stats.winsorization_metadata_winsor`

**Cause and message:**

<p>winsorization metadata requires winsor_n and both cap counts</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_resolved_lower"></span>

### `estimation.armstats.arm_stats.winsorization_resolved_lower`

**Cause and message:**

<p>winsorization resolved lower bound must be below upper bound</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.winsorization_upper_percentile"></span>

### `estimation.armstats.arm_stats.winsorization_upper_percentile`

**Cause and message:**

<p>winsorization upper percentile requires a resolved bound</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.x_role_declared"></span>

### `estimation.armstats.arm_stats.x_role_declared`

**Cause and message:**

<p>x_role must be declared when the x family is present</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.x_role_family"></span>

### `estimation.armstats.arm_stats.x_role_family`

**Cause and message:**

<p>x_role requires the x family to be present</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.arm_stats.x_role_one"></span>

### `estimation.armstats.arm_stats.x_role_one`

**Cause and message:**

<p>x_role must be one of &#x27;covariate&#x27;, &#x27;cluster_size&#x27;, or &#x27;uptake_total&#x27;; got {x_role!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `x_role`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.centered_raw_row"></span>

### `estimation.armstats.centered_raw_row`

**Cause and message:**

<p>centered/raw row carries unsupported weighted ArmStats fields: {unsupported}; weighted estimands must use ScoreStats.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unsupported`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.centered_sum_squares"></span>

### `estimation.armstats.centered_sum_squares`

**Cause and message:**

<p>{what}: centered sum of squares is {value:.6g}, negative beyond floating-point rounding (~{variance_slack:.3g}). Raw float64 sums lose the variance signal once spread/|mean| (the coefficient of variation) drops toward sqrt(eps) ~ 1e-8; but a deficit this large means the moment sums are inconsistent (mis-aggregated or corrupted upstream), not merely cancelled.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`, `variance_slack`, `what`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.cross_moment_materially"></span>

### `estimation.armstats.cross_moment_materially`

**Cause and message:**

<p>{what}: cross moment {cross:.6g} is materially nonzero against a zero variance (var_a={var_a:.6g}, var_b={var_b:.6g}), which no real sample can produce; the exact Cauchy-Schwarz bound is 0 and rounding at this scale is ~{floor:.3g}; these moments are corrupt, mis-aggregated, or drawn from mismatched partitions.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cross`, `floor`, `var_a`, `var_b`, `what`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.cross_moment_violates"></span>

### `estimation.armstats.cross_moment_violates`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/armstats.py#L1753).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `bound`, `cross`, `excess`, `var_a`, `var_b`, `variance_slack`, `what`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.independent_mean_contract"></span>

### `estimation.armstats.independent_mean_contract`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/armstats.py#L1830).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.score_stats.cluster_score_scale"></span>

### `estimation.armstats.score_stats.cluster_score_scale`

**Cause and message:**

<p>cluster_score_scale requires cluster_variance and n_clusters; a cluster scale without clustered moments has no meaning.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.score_stats.cluster_variance"></span>

### `estimation.armstats.score_stats.cluster_variance`

**Cause and message:**

<p>cluster_variance and n_clusters must be set together; a clustered second moment without its K has no coherent SE.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.score_stats.n_clusters_clustered"></span>

### `estimation.armstats.score_stats.n_clusters_clustered`

**Cause and message:**

<p>n_clusters must be &gt;= 1, got {n_clusters}; a clustered SE with no clusters has no coherent reference.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_clusters`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.score_stats.score_scale"></span>

### `estimation.armstats.score_stats.score_scale`

**Cause and message:**

<p>{field} must be finite and positive, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `value`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.score_stats.se_needs_positive"></span>

### `estimation.armstats.score_stats.se_needs_positive`

**Cause and message:**

<p>se() needs a positive normalizer, got {normalizer}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `normalizer`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.summary_stats.mean_finite"></span>

### `estimation.armstats.summary_stats.mean_finite`

**Cause and message:**

<p>mean must be finite, got {mean}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.armstats.summary_stats.var_finite"></span>

### `estimation.armstats.summary_stats.var_finite`

**Cause and message:**

<p>var must be finite, got {var}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `var`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.binary_provenance_required"></span>

### `estimation.binomial.binary_provenance_required`

**Cause and message:**

<p>binary_counts requires a declared conversion/retention metric type (a structural 0/1-per-unit fact), got metric_type={metric_type!r}: matching moments alone never proves a metric is genuinely Bernoulli</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.exact_counts_required"></span>

### `estimation.binomial.exact_counts_required`

**Cause and message:**

<p>metric {metric!r} is declared {declared_type!r}, but arm {group_id!r} carries no success count; declare the metric with the type its rows were built for, or rebuild the rows from unit data as a {declared_type} metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `declared_type`, `group_id`, `metric`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.finite_sample_arm_ceiling_exceeded"></span>

### `estimation.binomial.finite_sample_arm_ceiling_exceeded`

**Cause and message:**

<p>n_c={n_c!r}/n_t={n_t!r} exceed the {max_arm_size!r} units per arm the finite-sample conversion route is validated to: this is a compute-resource boundary, not a scientific one; per-call cost keeps growing with arm size, and this route refuses rather than spend unbounded compute on one estimate. A dense cell runs at any size with conversion_inference=&#x27;auto&#x27; (the default), which takes the delta-method interval; a sparse cell above this ceiling has no supported route</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `BinomialDataError`

**Context keys:** `max_arm_size`, `n_c`, `n_t`

**Registering modules:** `increment.estimation.binomial_rr`

<span id="estimation.binomial.finite_sample_unavailable"></span>

### `estimation.binomial.finite_sample_unavailable`

**Cause and message:**

<p>conversion_inference=&#x27;finite_sample&#x27; is unavailable for metric {metric!r}: {reason}. The finite-sample route is the independent-binomial test inversion for an unadjusted, unclustered, fixed-horizon conversion or retention contrast without an informative prior; use conversion_inference=&#x27;auto&#x27; (the default), or drop the incompatible setting</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `reason`

**Registering modules:** `increment.estimation.conversion_route`

<span id="estimation.binomial.inconsistent_bernoulli_variance"></span>

### `estimation.binomial.inconsistent_bernoulli_variance`

**Cause and message:**

<p>binary_counts(metric={metric!r}, group_id={group_id!r}): centered sum of squares cy2={cy2!r} does not match the Bernoulli-consistent value {expected_cy2!r} for n={n!r} within tolerance {tolerance!r}; this arm&#x27;s y values are not genuinely independent 0/1 draws (matching moments alone do not prove Bernoulli provenance; this is a corruption guard on top of the declared metric type)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `BinomialDataError`

**Context keys:** `cy2`, `expected_cy2`, `group_id`, `metric`, `n`, `tolerance`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.independent_units_required"></span>

### `estimation.binomial.independent_units_required`

**Cause and message:**

<p>binary_counts(metric={metric!r}, group_id={group_id!r}): the arm carries a CUPED covariate, ratio denominator, or uptake-mask family; clustered, weighted, adjusted, or otherwise non-unit-grain observations are not independent Bernoulli units this method admits</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.invalid_counts"></span>

### `estimation.binomial.invalid_counts`

**Cause and message:**

<p>binomial counts must satisfy 0 &lt;= x &lt;= n with integer n &gt;= 1, got x={x!r}, n={n!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `BinomialDataError`

**Context keys:** `n`, `x`

**Registering modules:** `increment.estimation.binomial_rr`

<span id="estimation.binomial.reconstructed_counts_not_binary"></span>

### `estimation.binomial.reconstructed_counts_not_binary`

**Cause and message:**

<p>binary_counts(metric={metric!r}, group_id={group_id!r}): success total {sum_y!r} over n={n!r} is not a valid integer count consistent with the binary moments</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `BinomialDataError`

**Context keys:** `group_id`, `metric`, `n`, `sum_y`

**Registering modules:** `increment.estimation.armstats`

<span id="estimation.binomial.tail_unrepresentable"></span>

### `estimation.binomial.tail_unrepresentable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/binomial_rr.py#L175).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `BinomialDataError`

**Context keys:** —

**Registering modules:** `increment.estimation.binomial_rr`

<span id="estimation.binomial.unknown_alternative"></span>

### `estimation.binomial.unknown_alternative`

**Cause and message:**

<p>Unknown alternative={alternative!r}: must be one of &quot;two-sided&quot;, &quot;greater&quot;, &quot;less&quot;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternative`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.binomial_rr`, `increment.estimation.encouragement`

<span id="estimation.cate.cate.covariate_missing_from"></span>

### `estimation.cate.cate.covariate_missing_from`

**Cause and message:**

<p>covariate {transform!r} is missing from the requested point; every interacted covariate that survived pruning is required: {names}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `names`, `transform`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.cate.level_contradicts_alpha"></span>

### `estimation.cate.cate.level_contradicts_alpha`

**Cause and message:**

<p>level={level!r} contradicts alpha={alpha!r}: the interval confidence implied by that alpha is {expected!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `expected`, `level`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.cluster_count"></span>

### `estimation.cate.cluster_count`

**Cause and message:**

<p>cluster count {n_clusters} exceeds the {n} fitted units</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `n_clusters`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.cluster_ids_shape"></span>

### `estimation.cate.cluster_ids_shape`

**Cause and message:**

<p>cluster_ids shape {shape} must match the {n} outcome rows</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `shape`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.cluster_weight"></span>

### `estimation.cate.cluster_weight`

**Cause and message:**

<p>cluster_weight must be &#x27;member_count&#x27; or &#x27;equal&#x27;, got {weighting!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `weighting`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate.categorical_so_knots"></span>

### `estimation.cate.covariate.categorical_so_knots`

**Cause and message:**

<p>covariate {self!r} is categorical, so knots do not apply: a one-hot basis already fits each level freely. Drop knots or declare it continuous</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `self`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate.needs_least_one"></span>

### `estimation.cate.covariate.needs_least_one`

**Cause and message:**

<p>covariate {name!r} needs at least one knot, got {knots}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `knots`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate.needs_strictly_increasing"></span>

### `estimation.cate.covariate.needs_strictly_increasing`

**Cause and message:**

<p>covariate {name!r} needs strictly increasing knots, got {knots}; repeated or unordered positions give duplicate hinge columns</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `knots`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate.non_finite_knots"></span>

### `estimation.cate.covariate.non_finite_knots`

**Cause and message:**

<p>covariate {name!r} has non-finite knots {knots}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `knots`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate.was_empty_knot"></span>

### `estimation.cate.covariate.was_empty_knot`

**Cause and message:**

<p>covariate {self!r} was given an empty knot tuple; pass None for no knots</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `self`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_columns_rows"></span>

### `estimation.cate.covariate_columns_rows`

**Cause and message:**

<p>covariate columns have {rows} rows, expected {n} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `rows`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_declared_both"></span>

### `estimation.cate.covariate_declared_both`

**Cause and message:**

<p>covariate {name!r} is declared both {prior_kind!r} and {kind!r}; pick one</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kind`, `name`, `prior_kind`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_declared_knots"></span>

### `estimation.cate.covariate_declared_knots`

**Cause and message:**

<p>covariate {name!r} is declared with knots {prior_knots!r} and {knots!r}; pick one; the adjustment and interaction blocks are built from a single basis per covariate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `knots`, `name`, `prior_knots`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_missing_from"></span>

### `estimation.cate.covariate_missing_from`

**Cause and message:**

<p>covariate {name!r} is missing from the unit frame; available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_nulls"></span>

### `estimation.cate.covariate_nulls`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/cate.py#L457).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_nulls_non"></span>

### `estimation.cate.covariate_nulls_non`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/cate.py#L452).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.covariate_shape"></span>

### `estimation.cate.covariate_shape`

**Cause and message:**

<p>covariate {name!r} must be 1-d, got shape {shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `shape`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.design.covariate_rows_expected"></span>

### `estimation.cate.design.covariate_rows_expected`

**Cause and message:**

<p>covariate {name!r} has {size} rows, expected {n}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `name`, `size`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.design.covariate_single_level"></span>

### `estimation.cate.design.covariate_single_level`

**Cause and message:**

<p>covariate {name!r} has zero variance in the fitting sample (single level {level!r}); drop it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `level`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.design.covariate_zero_variance"></span>

### `estimation.cate.design.covariate_zero_variance`

**Cause and message:**

<p>covariate {name!r} has zero variance in the fitting sample, so it cannot explain heterogeneity; drop it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.design_row_leverage"></span>

### `estimation.cate.design_row_leverage`

**Cause and message:**

<p>design row {row} has leverage {leverage:.6g}: it is fit exactly, so its HC2 weight is 0/0 and the reported standard error would be silently far too small. The columns pinning it are {culprits}; pool or drop them</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `culprits`, `leverage`, `row`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.design_too_wide"></span>

### `estimation.cate.design_too_wide`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/cate.py#L485).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `width`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.each_arm_needs"></span>

### `estimation.cate.each_arm_needs`

**Cause and message:**

<p>each arm needs at least 2 units; got {n_treated} treated and {n_control} control</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_control`, `n_treated`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.equal_weighting_without_cluster"></span>

### `estimation.cate.equal_weighting_without_cluster`

**Cause and message:**

<p>cluster_weight=&#x27;equal&#x27; requires cluster IDs</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.inference_metadata"></span>

### `estimation.cate.inference_metadata`

**Cause and message:**

<p>fit dimension, covariance and reference degrees must match the fitted design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.insufficient_clusters"></span>

### `estimation.cate.insufficient_clusters`

**Cause and message:**

<p>cluster uncertainty requires at least two declared clusters, got {count}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.intervention_grain"></span>

### `estimation.cate.intervention_grain`

**Cause and message:**

<p>intervention_grain must be &#x27;unit&#x27; or &#x27;cluster&#x27;, got {grain!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `grain`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.intervention_grain_without_cluster"></span>

### `estimation.cate.intervention_grain_without_cluster`

**Cause and message:**

<p>cluster intervention requires cluster identity and a cluster count</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.invalid_covariance"></span>

### `estimation.cate.invalid_covariance`

**Cause and message:**

<p>cluster covariance is unavailable: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.names_entries_gram"></span>

### `estimation.cate.names_entries_gram`

**Cause and message:**

<p>names has {count} entries for a {p}x{p} Gram matrix</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `p`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.outcome_constant_within"></span>

### `estimation.cate.outcome_constant_within`

**Cause and message:**

<p>outcome y is constant within both arms, so there is no sampling variation to report a standard error for</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.outcome_nulls_non"></span>

### `estimation.cate.outcome_nulls_non`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/cate.py#L435).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`, `increment.estimation.targeting`

<span id="estimation.cate.outcome_shape"></span>

### `estimation.cate.outcome_shape`

**Cause and message:**

<p>outcome y must be 1-d, got shape {shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `shape`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.protected_column_collinear"></span>

### `estimation.cate.protected_column_collinear`

**Cause and message:**

<p>protected column {name!r} is collinear with the columns before it (conditional variance {conditional:.3g} of diagonal {diag:.3g}); the design is degenerate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `conditional`, `diag`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.rank_deficient"></span>

### `estimation.cate.rank_deficient`

**Cause and message:**

<p>clustered design is rank deficient at dimension {dimension}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.score_nonfinite"></span>

### `estimation.cate.score_nonfinite`

**Cause and message:**

<p>Fitted scoring produced non-finite predictions for these covariates</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.score_state"></span>

### `estimation.cate.score_state`

**Cause and message:**

<p>Scoring state requires aligned finite coefficients and a valid fitted basis</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.single_cluster_direction"></span>

### `estimation.cate.single_cluster_direction`

**Cause and message:**

<p>{query} depends on a direction supported by only one cluster</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `query`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.singular_wald"></span>

### `estimation.cate.singular_wald`

**Cause and message:**

<p>interaction Wald test of dimension {dimension} has singular covariance; its uncertainty is unavailable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.thin_cells.one_hot_level_interacted_min"></span>

### `estimation.cate.thin_cells.one_hot_level_interacted_min`

**Cause and message:**

<p>one-hot level {name!r} holds {n_t} treated and {n_c} control unit(s); an interacted level needs at least 2 per arm. Below that the cell is fit exactly, the HC2 weight is 0/0, and the reported standard error is silently far too small. Pool the level or drop the interaction</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_c`, `n_t`, `name`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.thin_cells.one_hot_level_singleton"></span>

### `estimation.cate.thin_cells.one_hot_level_singleton`

**Cause and message:**

<p>one-hot level {name!r} holds {total} unit(s): a level that identifies a single row is fit exactly and its HC2 weight is 0/0. Pool it into another level or drop the covariate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `total`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.treatment_binary"></span>

### `estimation.cate.treatment_binary`

**Cause and message:**

<p>treatment d must be binary (0/1); got {values}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `values`

**Registering modules:** `increment.estimation.cate`, `increment.estimation.targeting`

<span id="estimation.cate.treatment_shape_expected"></span>

### `estimation.cate.treatment_shape_expected`

**Cause and message:**

<p>treatment d has shape {d_shape}, expected {y_shape} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `d_shape`, `y_shape`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.uncertainty_unavailable"></span>

### `estimation.cate.uncertainty_unavailable`

**Cause and message:**

<p>{query} has no finite positive uncertainty under the cluster sandwich</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `query`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.cate.zz_square_gram"></span>

### `estimation.cate.zz_square_gram`

**Cause and message:**

<p>zz must be a square Gram matrix, got shape {shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `shape`

**Registering modules:** `increment.estimation.cate`

<span id="estimation.certified.decimal_arithmetic_capacity"></span>

### `estimation.certified.decimal_arithmetic_capacity`

**Cause and message:**

<p>logarithm of {value!r} exceeds Decimal&#x27;s arithmetic capacity at precision {precision} (working precision {working_precision}, signal {signal})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `precision`, `signal`, `value`, `working_precision`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.decimal_input_exponent_capacity"></span>

### `estimation.certified.decimal_input_exponent_capacity`

**Cause and message:**

<p>rational input {value!r} exceeds Decimal&#x27;s exponent range at precision {precision} (working precision {working_precision})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `precision`, `value`, `working_precision`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.decimal_log_exponent_capacity"></span>

### `estimation.certified.decimal_log_exponent_capacity`

**Cause and message:**

<p>logarithm of {value!r} exceeds Decimal&#x27;s exponent range at precision {precision} (working precision {working_precision})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `precision`, `value`, `working_precision`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.decimal_precision_capacity"></span>

### `estimation.certified.decimal_precision_capacity`

**Cause and message:**

<p>logarithm exceeds Decimal&#x27;s precision capacity: requested precision {precision} needs working precision {working_precision}, which exceeds the {max_precision} limit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `max_precision`, `precision`, `working_precision`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.inexact_value"></span>

### `estimation.certified.inexact_value`

**Cause and message:**

<p>expected an exact Fraction or integer, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`

**Registering modules:** `increment.estimation._certified`, `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood`

<span id="estimation.certified.invalid_count"></span>

### `estimation.certified.invalid_count`

**Cause and message:**

<p>count must be a nonnegative integer</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`

**Registering modules:** `increment.estimation._certified`, `increment.estimation._sequential_likelihood`

<span id="estimation.certified.invalid_precision"></span>

### `estimation.certified.invalid_precision`

**Cause and message:**

<p>precision must be a positive integer</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `precision`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.nonpositive_gamma_value"></span>

### `estimation.certified.nonpositive_gamma_value`

**Cause and message:**

<p>gamma ratio requires a positive value</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.nonpositive_log_domain"></span>

### `estimation.certified.nonpositive_log_domain`

**Cause and message:**

<p>logarithm requires strictly positive bounds</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.nonpositive_rising_value"></span>

### `estimation.certified.nonpositive_rising_value`

**Cause and message:**

<p>rising factorial requires a positive value</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.nonpositive_root_width"></span>

### `estimation.certified.nonpositive_root_width`

**Cause and message:**

<p>max_width must be positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `max_width`

**Registering modules:** `increment.estimation._certified`, `increment.estimation._sequential_inversion`

<span id="estimation.certified.reversed_interval"></span>

### `estimation.certified.reversed_interval`

**Cause and message:**

<p>interval lower bound exceeds upper bound</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.reversed_root_domain"></span>

### `estimation.certified.reversed_root_domain`

**Cause and message:**

<p>lower bound exceeds upper bound</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.zero_divisor_interval"></span>

### `estimation.certified.zero_divisor_interval`

**Cause and message:**

<p>division by an interval containing zero</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.certified.zero_polynomial"></span>

### `estimation.certified.zero_polynomial`

**Cause and message:**

<p>the identically zero polynomial has no isolated roots</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `coefficients`

**Registering modules:** `increment.estimation._certified`

<span id="estimation.contrast.assignment_metadata"></span>

### `estimation.contrast.assignment_metadata`

**Cause and message:**

<p>{randomization_law} requires {expected_grain}, got {independence_grain}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected_grain`, `independence_grain`, `randomization_law`

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.contrast_interval_half"></span>

### `estimation.contrast.contrast_interval_half`

**Cause and message:**

<p>contrast interval half-width must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.contain_least_one"></span>

### `estimation.contrast.contrast_partition.contain_least_one`

**Cause and message:**

<p>contrast partition must contain at least one unit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.ct_counts"></span>

### `estimation.contrast.contrast_partition.ct_counts`

**Cause and message:**

<p>CT counts must be a mapping of nonnegative integers bounded by order draws</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.cycles_mapping"></span>

### `estimation.contrast.contrast_partition.cycles_mapping`

**Cause and message:**

<p>contrast partition cycles must be a mapping</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.cycles_positive_integers"></span>

### `estimation.contrast.contrast_partition.cycles_positive_integers`

**Cause and message:**

<p>contrast partition cycles must be positive integers</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.unit_deltas_cycles"></span>

### `estimation.contrast.contrast_partition.unit_deltas_cycles`

**Cause and message:**

<p>unit_deltas, cycles_by_unit and ct_counts_by_unit must have identical keys</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition.unit_deltas_finite"></span>

### `estimation.contrast.contrast_partition.unit_deltas_finite`

**Cause and message:**

<p>contrast partition unit deltas must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition_centered"></span>

### `estimation.contrast.contrast_partition_centered`

**Cause and message:**

<p>contrast partition centered moments must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partition_does"></span>

### `estimation.contrast.contrast_partition_does`

**Cause and message:**

<p>contrast partition {mismatch} does not match</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mismatch`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_partitions_disjoint"></span>

### `estimation.contrast.contrast_partitions_disjoint`

**Cause and message:**

<p>contrast partitions must be disjoint; unit overlap detected</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_reduction_delta_mean_finite"></span>

### `estimation.contrast.contrast_reduction_delta_mean_finite`

**Cause and message:**

<p>contrast reduction centered moments must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_reduction_least"></span>

### `estimation.contrast.contrast_reduction_least`

**Cause and message:**

<p>contrast reduction requires at least one partition</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats.control_group_treatment"></span>

### `estimation.contrast.contrast_stats.control_group_treatment`

**Cause and message:**

<p>control_group and treatment_group must be distinct</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_m2"></span>

### `estimation.contrast.contrast_stats_m2`

**Cause and message:**

<p>contrast stats M2 must be nonnegative</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_metric"></span>

### `estimation.contrast.contrast_stats_metric`

**Cause and message:**

<p>contrast stats metric {stats!r} does not match procedure metric {procedure!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `procedure`, `stats`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_n"></span>

### `estimation.contrast.contrast_stats_n`

**Cause and message:**

<p>contrast stats n_cycles must be positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_reconstructed"></span>

### `estimation.contrast.contrast_stats_reconstructed`

**Cause and message:**

<p>contrast stats reconstructed mean must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_reference"></span>

### `estimation.contrast.contrast_stats_reference`

**Cause and message:**

<p>contrast stats reference, residual, and M2 must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.contrast_stats_standard"></span>

### `estimation.contrast.contrast_stats_standard`

**Cause and message:**

<p>contrast stats standard error must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.legacy_envelope"></span>

### `estimation.contrast.legacy_envelope`

**Cause and message:**

<p>this switchback envelope result was saved before it recorded which side of the null its evidence rejects on; recompute it from the source data</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing_fields`, `remedy`

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.one_sided_alpha"></span>

### `estimation.contrast.one_sided_alpha`

**Cause and message:**

<p>one-sided alpha={procedure} doubles to alpha_eff={alpha_eff} &gt;= 1: no displayable two-sided interval exists at that level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha_eff`, `procedure`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.order_counts"></span>

### `estimation.contrast.order_counts`

**Cause and message:**

<p>CT and TC counts must sum to the declared number of order draws</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.procedure_alpha"></span>

### `estimation.contrast.procedure_alpha`

**Cause and message:**

<p>procedure alpha must be in (0, 1), got {procedure}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `procedure`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.procedure_contrastdecisionprocedure"></span>

### `estimation.contrast.procedure_contrastdecisionprocedure`

**Cause and message:**

<p>procedure must be ContrastDecisionProcedure, got {procedure}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `procedure`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.reference_metadata"></span>

### `estimation.contrast.reference_metadata`

**Cause and message:**

<p>contrast method, reference and degrees of freedom must match its replicates</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.replication_counts"></span>

### `estimation.contrast.replication_counts`

**Cause and message:**

<p>contrast counts must match the declared independent replicate grain</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.retained_window"></span>

### `estimation.contrast.retained_window`

**Cause and message:**

<p>retained_steps={retained_steps} must equal observation_steps={observation_steps} minus carryover_order={carryover_order} and be positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `carryover_order`, `observation_steps`, `retained_steps`

**Registering modules:** `increment.estimation.contrast_results`

<span id="estimation.contrast.shared_roster_size_mismatch"></span>

### `estimation.contrast.shared_roster_size_mismatch`

**Cause and message:**

<p>shared-block partitions must share one fixed roster size across every block; found {roster_sizes!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `roster_sizes`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.stats_contraststats"></span>

### `estimation.contrast.stats_contraststats`

**Cause and message:**

<p>stats must be ContrastStats, got {stats}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `stats`

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.switchback_block_inference"></span>

### `estimation.contrast.switchback_block_inference`

**Cause and message:**

<p>switchback block-t inference requires at least two independent blocks</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.contrast.switchback_unit_inference"></span>

### `estimation.contrast.switchback_unit_inference`

**Cause and message:**

<p>switchback unit-t inference requires at least two units</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.contrast`

<span id="estimation.crossfit.candidate_contribution_nonfinite"></span>

### `estimation.crossfit.candidate_contribution_nonfinite`

**Cause and message:**

<p>candidate {index} on fold {fold} returned {n_bad} non-finite contribution(s) out of {n_total}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fold`, `index`, `n_bad`, `n_total`, `route`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.candidate_returned_contribution"></span>

### `estimation.crossfit.candidate_returned_contribution`

**Cause and message:**

<p>candidate {index} returned a contribution of shape {shape} for a validation fold of {n_val} rows</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `index`, `n_val`, `shape`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.cluster_atomic_shape"></span>

### `estimation.crossfit.cluster_atomic_shape`

**Cause and message:**

<p>cluster_ids has shape {cluster_ids}, expected {assignment} to match the supplied assignment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `assignment`, `cluster_ids`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.cluster_ids_shape_expected"></span>

### `estimation.crossfit.cluster_ids_shape_expected`

**Cause and message:**

<p>cluster_ids has shape {cluster_ids}, expected {unit_ids} to match unit_ids</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster_ids`, `unit_ids`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.cluster_split"></span>

### `estimation.crossfit.cluster_split`

**Cause and message:**

<p>cluster {cluster!r} is split across the supplied assignment (values {values[0]!r} and {values[1]!r} both appear among its members); a cluster must land entirely on one side of a split, or entirely in one fold</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `values`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.fold_assignments_unit_ids_shape"></span>

### `estimation.crossfit.fold_assignments_unit_ids_shape`

**Cause and message:**

<p>unit_ids must be 1-d, got shape {unit_ids}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unit_ids`

**Registering modules:** `increment.estimation._adjust.overlap`, `increment.estimation.crossfit`

<span id="estimation.crossfit.identity_collision"></span>

### `estimation.crossfit.identity_collision`

**Cause and message:**

<p>{what} contains distinct native identities {first} and {second} that both render as {canonical!r}; canonicalize to one representation before calling rather than let them silently merge into one group</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `canonical`, `first`, `second`, `what`

**Registering modules:** `increment._identity`

<span id="estimation.crossfit.identity_missing"></span>

### `estimation.crossfit.identity_missing`

**Cause and message:**

<p>{what}[{index}] is missing (null/NaN); every row needs an identity</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `index`, `what`

**Registering modules:** `increment._identity`

<span id="estimation.crossfit.insufficient_clusters"></span>

### `estimation.crossfit.insufficient_clusters`

**Cause and message:**

<p>only {n_clusters} atomic group(s) available in a stratum for n_folds={n_folds}; cross-fitting needs at least one group per stratum per fold; groups are distinct cluster_ids when supplied, otherwise distinct unit_ids; lower n_folds or supply more groups</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_clusters`, `n_folds`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.n_candidates_least"></span>

### `estimation.crossfit.n_candidates_least`

**Cause and message:**

<p>n_candidates must be at least 1, got {n_candidates}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_candidates`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.n_folds_least"></span>

### `estimation.crossfit.n_folds_least`

**Cause and message:**

<p>n_folds must be at least 2, got {n_folds}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_folds`

**Registering modules:** `increment.estimation.crossfit`, `increment.estimation.targeting`

<span id="estimation.crossfit.selection_needs_least"></span>

### `estimation.crossfit.selection_needs_least`

**Cause and message:**

<p>selection needs at least 2 fold labels, got {n_labels}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_labels`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.stratify_shape_expected"></span>

### `estimation.crossfit.stratify_shape_expected`

**Cause and message:**

<p>stratify has shape {s}, expected ({n},) to match unit_ids</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `s`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.crossfit.test_size"></span>

### `estimation.crossfit.test_size`

**Cause and message:**

<p>test_size must be in (0, 1), got {test_size}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `test_size`

**Registering modules:** `increment.estimation.crossfit`

<span id="estimation.cuped.arm_no_covariate"></span>

### `estimation.cuped.arm_no_covariate`

**Cause and message:**

<p>Arm {arm_group}/{arm_metric} has no covariate data (ref_x/cx1/cx2/cxy is None). The covariate must be materialised before requesting cuped: from a definitions path, set n_pre_periods &gt; 0 on the experiment; from from_unit_summary, declare the column with Method.cuped(covariates=...).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_group`, `arm_metric`

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped.covariate_zero_variance"></span>

### `estimation.cuped.covariate_zero_variance`

**Cause and message:**

<p>Covariate has zero variance (inverse-n weighted within-arm variance={weighted_var_x}). Cannot compute CUPED theta. Ensure the covariate is not constant; if it varies but sits at a large offset (|mean| &gt;&gt; spread), floating-point cancellation destroyed its variance before this seam; centre the covariate (subtract a constant) before aggregation.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `weighted_var_x`

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped.cuped_adjusted_variance"></span>

### `estimation.cuped.cuped_adjusted_variance`

**Cause and message:**

<p>CUPED-adjusted variance for arm {arm_group!r}/{arm_metric!r} is {naive_var:.6g}, negative beyond floating-point rounding; the slope implies an impossible {pair} correlation for this arm (Cauchy-Schwarz: {cross_label} cannot exceed sqrt(Var(Y)*Var(X))); the moments are corrupt or were not produced together</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_group`, `arm_metric`, `cross_label`, `naive_var`, `pair`

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped.pooled_theta_least"></span>

### `estimation.cuped.pooled_theta_least`

**Cause and message:**

<p>pooled_theta requires at least 2 arms (control + treatment)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped.ratio_arm_covariate_role"></span>

### `estimation.cuped.ratio_arm_covariate_role`

**Cause and message:**

<p>Arm {arm_group}/{arm_metric} declares x_role={x_role!r}, so its x family is not a CUPED covariate and its cross moment with the denominator means something else. Ratio CUPED needs x_role=&#x27;covariate&#x27;.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_group`, `arm_metric`, `x_role`

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped.ratio_arm_no_denominator_cross"></span>

### `estimation.cuped.ratio_arm_no_denominator_cross`

**Cause and message:**

<p>Arm {arm_group}/{arm_metric} carries a covariate and a ratio denominator but no cross moment between them (cxden is None), so the denominator&#x27;s CUPED slope is not identified. Ratio CUPED needs Cov(covariate, denominator); supply moments in the centered (format-2) shape, which carries cxden, rather than the raw-sum (format-1) shape, which has no column for it.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_group`, `arm_metric`

**Registering modules:** `increment.estimation.cuped.scalar`

<span id="estimation.cuped_linear.arm_counts"></span>

### `estimation.cuped_linear.arm_counts`

**Cause and message:**

<p>each arm needs at least two units; arm counts are {counts!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `counts`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.covariate_input"></span>

### `estimation.cuped_linear.covariate_input`

**Cause and message:**

<p>covariate declaration/input names differ: values={values!r}, kinds={kinds!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kinds`, `values`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.covariate_kind"></span>

### `estimation.cuped_linear.covariate_kind`

**Cause and message:**

<p>covariate {name!r} has unsupported kind {kind!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kind`, `name`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.incompatible_evidence"></span>

### `estimation.cuped_linear.incompatible_evidence`

**Cause and message:**

<p>incompatible CUPED compact evidence: reason={reason!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.input_shape"></span>

### `estimation.cuped_linear.input_shape`

**Cause and message:**

<p>{name} must be one-dimensional and align with {n} arm labels; got shape {shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `name`, `shape`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.invalid_alpha"></span>

### `estimation.cuped_linear.invalid_alpha`

**Cause and message:**

<p>alpha must be strictly between zero and one; got {alpha}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.invalid_arms"></span>

### `estimation.cuped_linear.invalid_arms`

**Cause and message:**

<p>arm labels/order/control disagree: control={control!r}, order={arm_order!r}, observed={observed!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_order`, `control`, `observed`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.invalid_evidence"></span>

### `estimation.cuped_linear.invalid_evidence`

**Cause and message:**

<p>invalid CUPED compact evidence: reason={reason!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.invalid_scale"></span>

### `estimation.cuped_linear.invalid_scale`

**Cause and message:**

<p>projection scale must be &#x27;log&#x27; or &#x27;absolute&#x27;, got {scale!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `scale`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.level_support"></span>

### `estimation.cuped_linear.level_support`

**Cause and message:**

<p>categorical level for {column!r} has fewer than two units in an arm: level={level!r}, counts={counts!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `counts`, `level`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.leverage"></span>

### `estimation.cuped_linear.leverage`

**Cause and message:**

<p>unit leverage is too close to one: row={row}, leverage={leverage}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `leverage`, `row`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.merge_precision"></span>

### `estimation.cuped_linear.merge_precision`

**Cause and message:**

<p>merged categorical evidence cannot preserve the {component} mean of arm {arm!r}: its error bound {bound!r} exceeds 2**-36 of the mean {mean!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `bound`, `component`, `mean`, `route`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.missing_refused"></span>

### `estimation.cuped_linear.missing_refused`

**Cause and message:**

<p>covariate {name!r} contains missing values under missing=&#x27;refuse&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.nonfinite"></span>

### `estimation.cuped_linear.nonfinite`

**Cause and message:**

<p>outcome component {name!r} contains a non-finite value at row {row}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `row`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.nonfinite_projection"></span>

### `estimation.cuped_linear.nonfinite_projection`

**Cause and message:**

<p>projection produced non-finite numerical evidence: value={value}, se={se}, dof={dof}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dof`, `se`, `value`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.nonpositive_component"></span>

### `estimation.cuped_linear.nonpositive_component`

**Cause and message:**

<p>adjusted component {family!r} for arm {arm!r} must be positive for a log or ratio projection; got {value}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `family`, `value`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.outcome_components"></span>

### `estimation.cuped_linear.outcome_components`

**Cause and message:**

<p>supply either y or both numerator and denominator components</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.projection_arm"></span>

### `estimation.cuped_linear.projection_arm`

**Cause and message:**

<p>projection requires distinct declared treatment/control arms: {arm!r}, {control!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `control`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.projection_kind"></span>

### `estimation.cuped_linear.projection_kind`

**Cause and message:**

<p>projection {requested!r} is incompatible with fit kind {available!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `requested`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.projection_unavailable"></span>

### `estimation.cuped_linear.projection_unavailable`

**Cause and message:**

<p>projection is unavailable for fit: kind={kind!r}, arm={arm!r}, control={control!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `control`, `kind`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.rank_deficiency"></span>

### `estimation.cuped_linear.rank_deficiency`

**Cause and message:**

<p>encoded covariates are rank deficient: columns={columns!r}, partners={partners!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `partners`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.residual_df"></span>

### `estimation.cuped_linear.residual_df`

**Cause and message:**

<p>residual degrees of freedom must be positive; got {df}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `df`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.cuped_linear.support_region"></span>

### `estimation.cuped_linear.support_region`

**Cause and message:**

<p>fit is outside support region {region!r}: violations={violations!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `region`, `violations`

**Registering modules:** `increment.estimation.cuped._numeric`

<span id="estimation.diagnostics.allocation_band_ds"></span>

### `estimation.diagnostics.allocation_band_ds`

**Cause and message:**

<p>allocation band (ds={ds!r}, group_id={group_id!r}) is [{lower!r}, {upper!r}], which touches 0 or 1; at credible_level={credible_level!r} this posterior (Beta({a}, {b})) cannot resolve a proper interval, and a bound of exactly 0 or 1 would read as certainty about the allocation; request a less extreme credible_level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `a`, `b`, `credible_level`, `ds`, `group_id`, `lower`, `upper`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.allocation_posterior_bands"></span>

### `estimation.diagnostics.allocation_posterior_bands`

**Cause and message:**

<p>allocation_posterior_bands computes bands for one experiment at a time (it pools n_total per ds across every row it sees), but got rows spanning multiple experiment_id values: {experiment_ids!r}. Filter to a single experiment before calling.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment_ids`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.alpha"></span>

### `estimation.diagnostics.alpha`

**Cause and message:**

<p>alpha must be in (0, 1), got {alpha}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.breakout.estimates`, `increment.estimation._fixed_horizon`, `increment.estimation.absorption`, `increment.estimation.cate`, `increment.estimation.diagnostics`, `increment.estimation.encouragement`, `increment.estimation.inference`, `increment.estimation.quantile`, `increment.estimation.targeting`, `increment.power.core`

<span id="estimation.diagnostics.always_srm_predeclared"></span>

### `estimation.diagnostics.always_srm_predeclared`

**Cause and message:**

<p>inference=&#x27;always_valid&#x27; requires predeclared allocation support; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `route`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.counts_missing_columns"></span>

### `estimation.diagnostics.counts_missing_columns`

**Cause and message:**

<p>counts missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.counts_non_negative"></span>

### `estimation.diagnostics.counts_non_negative`

**Cause and message:**

<p>counts must be non-negative, got {negative}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `negative`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.counts_whole_unit"></span>

### `estimation.diagnostics.counts_whole_unit`

**Cause and message:**

<p>counts must be whole unit counts, got fractional values {fractional}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fractional`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.credible_level"></span>

### `estimation.diagnostics.credible_level`

**Cause and message:**

<p>credible_level must be in (0, 1), got {credible_level!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `credible_level`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.credible_level_too"></span>

### `estimation.diagnostics.credible_level_too`

**Cause and message:**

<p>credible_level={credible_level!r} is too close to 1 to resolve a tail probability in floating point; request a less extreme credible_level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `credible_level`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.duplicate_ds_group"></span>

### `estimation.diagnostics.duplicate_ds_group`

**Cause and message:**

<p>duplicate (ds, group_id) row {ds_group!r}: n_total would double-count this arm and every band that day would be wrong</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `ds_group`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.every_expected_share_finite"></span>

### `estimation.diagnostics.every_expected_share_finite`

**Cause and message:**

<p>every expected share must be finite, got {nonfinite}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `nonfinite`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.every_expected_share_positive"></span>

### `estimation.diagnostics.every_expected_share_positive`

**Cause and message:**

<p>every expected share must be &gt; 0; a zero-share arm makes the chi-square statistic undefined for any observed count there (and negative shares are meaningless); got {expected}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.expected_allocation_contain"></span>

### `estimation.diagnostics.expected_allocation_contain`

**Cause and message:**

<p>expected allocation cannot contain accounting labels {accounting}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `accounting`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.expected_keys_do"></span>

### `estimation.diagnostics.expected_keys_do`

**Cause and message:**

<p>expected keys {expected_keys} do not match counts keys {counts_keys}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `counts_keys`, `expected_keys`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.expected_share_normalization"></span>

### `estimation.diagnostics.expected_share_normalization`

**Cause and message:**

<p>expected-share normalization produced nonpositive or nonfinite shares; got {invalid_normalized}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `invalid_normalized`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.inference_always_valid"></span>

### `estimation.diagnostics.inference_always_valid`

**Cause and message:**

<p>inference must be &#x27;always_valid&#x27; or &#x27;fixed&#x27;, got {inference!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `inference`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.n_cumulative_non"></span>

### `estimation.diagnostics.n_cumulative_non`

**Cause and message:**

<p>n_cumulative must be non-negative, got {n_raw!r} for (ds={ds!r}, group_id={group_id!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `ds`, `group_id`, `n_raw`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.n_cumulative_whole"></span>

### `estimation.diagnostics.n_cumulative_whole`

**Cause and message:**

<p>n_cumulative must be a whole unit count, got {n_raw!r} for (ds={ds!r}, group_id={group_id!r}); silently truncating would misstate the posterior</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `ds`, `group_id`, `n_raw`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.need_least_groups"></span>

### `estimation.diagnostics.need_least_groups`

**Cause and message:**

<p>Need at least 2 groups for SRM test</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.predeclared_allocation_support"></span>

### `estimation.diagnostics.predeclared_allocation_support`

**Cause and message:**

<p>predeclared allocation support via expected is required for both fixed and always_valid inference; always_valid validity also requires known, constant conditional assignment probabilities</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.prior_alpha_finite"></span>

### `estimation.diagnostics.prior_alpha_finite`

**Cause and message:**

<p>prior alpha must be finite and positive, got {prior_a!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `prior_a`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.prior_beta_finite"></span>

### `estimation.diagnostics.prior_beta_finite`

**Cause and message:**

<p>prior beta must be finite and positive, got {prior_b!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `prior_b`

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.diagnostics.total_observed_count"></span>

### `estimation.diagnostics.total_observed_count`

**Cause and message:**

<p>Total observed count must be &gt; 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.diagnostics`

<span id="estimation.encouragement.alpha_eff_too"></span>

### `estimation.encouragement.alpha_eff_too`

**Cause and message:**

<p>alpha_eff is too small: alpha_eff / 2 underflows floating-point precision</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.encouragement`

<span id="estimation.encouragement.cluster.arm_needs_two"></span>

### `estimation.encouragement.cluster.arm_needs_two`

**Cause and message:**

<p>metric &#x27;{metric}&#x27;: cluster contrast (&#x27;{cluster}&#x27;) needs at least 2 clusters in EACH arm to estimate a between-cluster variance for that arm, got {k_t} treatment cluster(s) and {k_c} control cluster(s); a single cluster carries no between-cluster variance contribution to estimate its own arm&#x27;s component from.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `k_c`, `k_t`, `metric`

**Registering modules:** `increment.estimation.encouragement`, `increment.estimation.engine`

<span id="estimation.encouragement.cluster.ratio"></span>

### `estimation.encouragement.cluster.ratio`

**Cause and message:**

<p>ratio metric(s) {names} are not supported with a declared cluster (&#x27;{cluster}&#x27;) under an encouragement design; the first stage reads cluster SIZES out of the den family a clustered ratio metric needs for its own denominator, so the two cannot share one row. Analyze the ratio metric without the encouragement design, or without the cluster.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `names`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.control.missing"></span>

### `estimation.encouragement.control.missing`

**Cause and message:**

<p>control_group &#x27;{control_group}&#x27; not found in summary</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_group`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.cuped_adjusted_late"></span>

### `estimation.encouragement.cuped_adjusted_late`

**Cause and message:**

<p>CUPED-adjusted LATE needs the covariate x uptake cross-moment cxd, which is None for arm {arm_group}/{arm_metric} although its covariate moments are present; the moments were aggregated before this estimator existed. Re-aggregate the summary (re-run the query, or rebuild from_unit_summary with a CUPED method declaring covariates and uptake=&#x27;&lt;column&gt;&#x27;), or request the late estimand with variance_reduction=&#x27;none&#x27;.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_group`, `arm_metric`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.dof_positive_reference"></span>

### `estimation.encouragement.dof_positive_reference`

**Cause and message:**

<p>dof must be positive for a t reference, got {dof}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dof`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.encouragement`

<span id="estimation.encouragement.late.ratio"></span>

### `estimation.encouragement.late.ratio`

**Cause and message:**

<p>LATE is not supported for ratio metric &#x27;{metric}&#x27;; the Wald numerator is the difference of arm mean_y, which for a ratio metric is the NUMERATOR mean, so the result would be the numerator&#x27;s LATE silently mislabelled as the ratio&#x27;s. Request estimands=(&#x27;itt&#x27;,) (ratio-aware), or model the numerator and denominator as separate mean metrics</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.needs_uptake_cyd"></span>

### `estimation.encouragement.needs_uptake_cyd`

**Cause and message:**

<p>needs uptake: cyd is None (uptake fact not materialised)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.negative_variance"></span>

### `estimation.encouragement.negative_variance`

**Cause and message:**

<p>{what} computed as {value:.6g}, negative beyond floating-point rounding (~{slack:.3g}); this scale of deficit means the upstream moments feeding the Wald ratio&#x27;s delta-method variance are infeasible or inconsistent (mis-aggregated or corrupted), not merely cancelled.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `slack`, `value`, `what`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.one_sided_encouragement"></span>

### `estimation.encouragement.one_sided_encouragement`

**Cause and message:**

<p>one_sided encouragement declared but control arm has uptake (uptake_total={control_uptake_total:g}, metric &#x27;{metric}&#x27;); control must be structurally unable to take up; this is an instrumentation bug, refusing to estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_uptake_total`, `metric`

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.prior_cluster_robust"></span>

### `estimation.encouragement.prior_cluster_robust`

**Cause and message:**

<p>prior and a cluster-robust t reference are mutually exclusive: a Normal prior has no coherent conjugate update against a t sampling distribution</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.prior_inference_are"></span>

### `estimation.encouragement.prior_inference_are`

**Cause and message:**

<p>prior and inference are mutually exclusive: a conjugate posterior has no always-valid guarantee under repeated looks</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.sequential_inference_cluster"></span>

### `estimation.encouragement.sequential_inference_cluster`

**Cause and message:**

<p>sequential inference and a cluster-robust t reference are mutually exclusive; the sequential boundaries assume independent units at unit grain</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.encouragement`

<span id="estimation.encouragement.unknown_estimand_supported"></span>

### `estimation.encouragement.unknown_estimand_supported`

**Cause and message:**

<p>unknown estimand(s) {unknown}; supported: {estimands}. As-treated / per-protocol comparisons are deliberately not offered.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimands`, `unknown`

**Registering modules:** `increment.estimation.encouragement`, `increment.readouts._common`

<span id="estimation.engine.arm.duplicate_rows"></span>

### `estimation.engine.arm.duplicate_rows`

**Cause and message:**

<p>group_summary carries more than one row for (metric, group_id) {keys}; an unchecked duplicate control row silently overwrites an earlier one in the lookup dict, and an unchecked duplicate treatment row is silently iterated as if it were a second arm, either way returning an arbitrary lift instead of a refusal. Deduplicate group_summary to exactly one row per (metric, group_id) before calling estimate_lift.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `keys`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.arm.invalid_count"></span>

### `estimation.engine.arm.invalid_count`

**Cause and message:**

<p>{what} is {value!r}, not a finite integer-valued count; ArmStats.n rejects a fractional value at direct construction, so the untyped dataframe ingress must too instead of silently truncating it (int(2.9) == 2) into a different arm size than the row declared. Round or fix the upstream aggregation so &#x27;n&#x27; is integral.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`, `what`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.carries_format_moments"></span>

### `estimation.engine.carries_format_moments`

**Cause and message:**

<p>{what} carries format-1 moments (raw additive sums: &#x27;sum_y2&#x27; present, &#x27;cy2&#x27; absent). increment consumes CENTERED moments: ref_y/cy1/cy2 and the c-form families; raw second moments lose the variance signal to floating-point cancellation once the mean dwarfs the spread, so they are never reinterpreted here. Two ways in: from_moments(...), which adapts stamped or unstamped format-1 rows automatically, or ArmStats.from_raw_sums(...) for hand-built rows.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `what`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.carries_unsupported_weighted"></span>

### `estimation.engine.carries_unsupported_weighted`

**Cause and message:**

<p>{what} carries unsupported weighted ArmStats fields: {unsupported}; weighted estimands must use ScoreStats.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unsupported`, `what`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cluster.denominator"></span>

### `estimation.engine.cluster.denominator`

**Cause and message:**

<p>metric &#x27;{metric}&#x27; (type &#x27;{metric_type}&#x27;, contrast &#x27;{treatment_group}&#x27; vs &#x27;{control_group}&#x27;) carries a populated denominator family (ref_den) with no cluster= declared; this group_summary was built with cluster=... (so &#x27;n&#x27; counts clusters, not units), and reading it here as unit grain would silently mislabel a cluster-level point estimate and SE as unit-level. Pass estimate_lift(cluster=&lt;the same column&gt;, ...).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `control_group`, `metric`, `metric_type`, `treatment_group`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.control_group_found"></span>

### `estimation.engine.control_group_found`

**Cause and message:**

<p>control_group &#x27;{control_group}&#x27; not found in arms. Available groups: {known_groups}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_group`, `known_groups`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_cluster"></span>

### `estimation.engine.cuped_linear_cluster`

**Cause and message:**

<p>multicovariate CUPED does not support clustered assignment (cluster={cluster!r}); choose an unadjusted clustered method or scalar CUPED where supported.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `cluster`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_encouragement"></span>

### `estimation.engine.cuped_linear_encouragement`

**Cause and message:**

<p>multicovariate CUPED is not available for encouragement design metric {metric!r}; use an encouragement-supported method without CUPED.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_observational"></span>

### `estimation.engine.cuped_linear_observational`

**Cause and message:**

<p>multicovariate CUPED is a randomized mean/ratio estimator and cannot be used with observational metric {metric!r}; choose an observational estimator or a randomized design.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_sequential"></span>

### `estimation.engine.cuped_linear_sequential`

**Cause and message:**

<p>multicovariate CUPED is fixed-horizon only and cannot be used with sequential inference; use an unadjusted sequential method or scalar CUPED&#x27;s registered sequential route.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_unit_evidence"></span>

### `estimation.engine.cuped_linear_unit_evidence`

**Cause and message:**

<p>multicovariate CUPED for metric {metric!r} requires per-unit outcomes and covariates declared on the method; use a per-unit analysis entry point instead of aggregate moments.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.cuped_linear_winsorization"></span>

### `estimation.engine.cuped_linear_winsorization`

**Cause and message:**

<p>multicovariate CUPED cannot be composed with winsorization for metric {metric!r}; choose one supported construction.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.group_summary_missing"></span>

### `estimation.engine.group_summary_missing`

**Cause and message:**

<p>group_summary missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.group_summary_row"></span>

### `estimation.engine.group_summary_row`

**Cause and message:**

<p>group_summary row missing required columns: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.method.name_iptw_does"></span>

### `estimation.engine.method.name_iptw_does`

**Cause and message:**

<p>Method(name=&#x27;iptw&#x27;) does not accept outcome_learner/folds; IPTW fits one propensity model with no cross-fitting; set propensity_learner (mapped to iptw_estimate&#x27;s learner=) instead, or use Method(name=&#x27;dml&#x27;) / Method(name=&#x27;aipw&#x27;) for cross-fit outcome-model pluggability.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.declarations`

<span id="estimation.engine.method.name_without_variance"></span>

### `estimation.engine.method.name_without_variance`

**Cause and message:**

<p>Method(name={self!r}) without its matching variance_reduction would label an unadjusted estimate as CUPED-adjusted.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `self`

**Registering modules:** `increment.declarations`

<span id="estimation.engine.method_covariates"></span>

### `estimation.engine.method_covariates`

**Cause and message:**

<p>method {method!r} has an invalid CUPED covariate declaration for its variance-reduction procedure</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `method`

**Registering modules:** `increment.declarations`

<span id="estimation.engine.method_name_observational"></span>

### `estimation.engine.method_name_observational`

**Cause and message:**

<p>Method(name={m!r}) is an observational adjustment (estimate_ate / Observational design); this randomized path would label an unadjusted randomized estimate with it.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `m`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.method_names_unique"></span>

### `estimation.engine.method_names_unique`

**Cause and message:**

<p>{caller}: method names must be unique within one metric request; duplicates: {duplicates!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `duplicates`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.metric_found_group"></span>

### `estimation.engine.metric_found_group`

**Cause and message:**

<p>Metric(s) {unknown} found in group_summary but not in the metrics list. Declared metrics: {metric_types}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_types`, `unknown`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.ratio.cuped"></span>

### `estimation.engine.ratio.cuped`

**Cause and message:**

<p>CUPED cannot be applied to ratio metric &#x27;{metric}&#x27; from this source: experiment &#x27;{experiment}&#x27; declares n_pre_periods=0, and the warehouse covariate for a ratio metric is its numerator&#x27;s pre-period total, so no covariate moments (ref_x/cx1/cx2/cxy) or covariate-denominator cross moment (cxden) can be materialized. Declare n_pre_periods &gt; 0 on the experiment, or supply the covariate through Method.cuped(covariates=...) in the frame&#x27;s MetricSpec method declaration.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `experiment`, `metric`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.ratio.negative_variance"></span>

### `estimation.engine.ratio.negative_variance`

**Cause and message:**

<p>ratio_abs_diff_se refused these moments: {reason}; a deficit at this scale means the numerator/denominator moments violate Cauchy-Schwarz (an infeasible covariance), not merely cancelled, so reporting a clamped zero-SE (maximal confidence) would be the most dangerous possible failure mode for a decision number.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.winsorization_metadata_differs"></span>

### `estimation.engine.winsorization_metadata_differs`

**Cause and message:**

<p>winsorization metadata differs between arms in {field}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.winsorization_metadata_present"></span>

### `estimation.engine.winsorization_metadata_present`

**Cause and message:**

<p>winsorization metadata must be present on both contrast arms</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.engine`

<span id="estimation.engine.x_role_declared"></span>

### `estimation.engine.x_role_declared`

**Cause and message:**

<p>x_role must be declared when the x family is present; an explicit null x_role is not a legacy declaration</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.engine`

<span id="estimation.estimator.adjustment.callback_contract"></span>

### `estimation.estimator.adjustment.callback_contract`

**Cause and message:**

<p>Adjustment callback for {estimator_id!r} must accept AdjustmentInputs, Metric and Design, plus the shared adjustment keyword arguments.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.estimator.implementation.missing_ref"></span>

### `estimation.estimator.implementation.missing_ref`

**Cause and message:**

<p>Implementation for estimator {estimator_id!r} has no declared estimator ref.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.estimator.need.unsupported"></span>

### `estimation.estimator.need.unsupported`

**Cause and message:**

<p>Estimator {estimator_id!r} declares unsupported need spec {need_type!r}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`, `need_type`

**Registering modules:** `increment.estimators.registry`

<span id="estimation.estimator.ref.already_registered"></span>

### `estimation.estimator.ref.already_registered`

**Cause and message:**

<p>Estimator {estimator_id!r} is already registered with a different declaration.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.estimator.ref.invalid_id"></span>

### `estimation.estimator.ref.invalid_id`

**Cause and message:**

<p>Estimator id {estimator_id!r} must be a qualified identifier.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.estimator.ref.observational_needs"></span>

### `estimation.estimator.ref.observational_needs`

**Cause and message:**

<p>Observational estimator {estimator_id!r} needs a route with total moments and design rows.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.estimator.ref.raw_rows_unsupported"></span>

### `estimation.estimator.ref.raw_rows_unsupported`

**Cause and message:**

<p>Observational estimator {estimator_id!r} declares raw unit rows, but observational estimators receive transformed rows; declare UnitRowsSpec(outcome_stage=&#x27;transformed&#x27;).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimator_id`

**Registering modules:** `increment.declarations`

<span id="estimation.family.bh_select_q_finite"></span>

### `estimation.family.bh_select_q_finite`

**Cause and message:**

<p>q must be finite and in (0, 1], got {q!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `q`

**Registering modules:** `increment.estimation.family`

<span id="estimation.family.e_values_finite"></span>

### `estimation.family.e_values_finite`

**Cause and message:**

<p>log_e_values[{i}] must be an exact lower log bound or tagged infinity, got {e!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `e`, `i`

**Registering modules:** `increment.estimation.family`

<span id="estimation.family.exploratory_construction"></span>

### `estimation.family.exploratory_construction`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/exploratory_family.py#L92).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `constructions`, `rows`

**Registering modules:** `increment.estimation.exploratory_family`

<span id="estimation.family.exploratory_non_decision"></span>

### `estimation.family.exploratory_non_decision`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/exploratory_family.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `rows`

**Registering modules:** `increment.estimation.exploratory_family`

<span id="estimation.family.exploratory_pre_corrected"></span>

### `estimation.family.exploratory_pre_corrected`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/exploratory_family.py#L67).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `rows`

**Registering modules:** `increment.estimation.exploratory_family`

<span id="estimation.family.exploratory_row"></span>

### `estimation.family.exploratory_row`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/exploratory_family.py#L58).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `rows`

**Registering modules:** `increment.estimation.exploratory_family`

<span id="estimation.family.exploratory_sequential"></span>

### `estimation.family.exploratory_sequential`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/exploratory_family.py#L83).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `rows`

**Registering modules:** `increment.estimation.exploratory_family`

<span id="estimation.family.nominal_alpha_finite"></span>

### `estimation.family.nominal_alpha_finite`

**Cause and message:**

<p>nominal_alpha must be finite and in (0, 1), got {nominal_alpha!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `nominal_alpha`

**Registering modules:** `increment.estimation.family`

<span id="estimation.family.p_values_finite"></span>

### `estimation.family.p_values_finite`

**Cause and message:**

<p>p_values[{i}] must be finite and in [0, 1], got {p!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `i`, `p`

**Registering modules:** `increment.estimation.family`

<span id="estimation.health.check.available_conclusion"></span>

### `estimation.health.check.available_conclusion`

**Cause and message:**

<p>available checks require a diagnostic conclusion</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.health`

<span id="estimation.health.check.unavailable_conclusion"></span>

### `estimation.health.check.unavailable_conclusion`

**Cause and message:**

<p>unavailable and not-applicable checks cannot have a conclusion</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.health`

<span id="estimation.health.check.unavailable_reason"></span>

### `estimation.health.check.unavailable_reason`

**Cause and message:**

<p>unavailable and not-applicable checks require a reason in context</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.health`

<span id="estimation.health.experiment.checks_type"></span>

### `estimation.health.experiment.checks_type`

**Cause and message:**

<p>checks must be a tuple or list, got {received}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `received`

**Registering modules:** `increment.estimation.health`

<span id="estimation.health.experiment.stable_order"></span>

### `estimation.health.experiment.stable_order`

**Cause and message:**

<p>health checks must use stable check/population/metric/feed order</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.health`

<span id="estimation.inference.abs_dof_requires_dof"></span>

### `estimation.inference.abs_dof_requires_dof`

**Cause and message:**

<p>abs_dof is only meaningful alongside a cluster-robust dof (the absolute sidecar&#x27;s own reference df exists only when the primary contrast is already on the cluster-robust t-reference path); got abs_dof set with dof=None.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.arm_ns_each"></span>

### `estimation.inference.arm_ns_each`

**Cause and message:**

<p>arm_ns must each be &gt; 1 for a Welch-Satterthwaite df, got {arm_ns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_ns`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.degenerate_data_zero"></span>

### `estimation.inference.degenerate_data_zero`

**Cause and message:**

<p>Degenerate data: zero variance, cannot estimate lift uncertainty</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.infer_ate_null_abs_finite"></span>

### `estimation.inference.infer_ate_null_abs_finite`

**Cause and message:**

<p>null_abs must be finite, got {null_abs}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_abs`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.infer_ate_null_abs_on_absolute_metric"></span>

### `estimation.inference.infer_ate_null_abs_on_absolute_metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/inference.py#L118).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.infer_ate_null_lift_finite"></span>

### `estimation.inference.infer_ate_null_lift_finite`

**Cause and message:**

<p>null_lift must be finite, got {null_lift}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.infer_ate_null_lift_on_absolute_metric"></span>

### `estimation.inference.infer_ate_null_lift_on_absolute_metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/inference.py#L123).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.infer_lift_abs_diff_finite"></span>

### `estimation.inference.infer_lift_abs_diff_finite`

**Cause and message:**

<p>abs_diff must be finite, got {abs_diff}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `abs_diff`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.infer_lift_abs_se_finite"></span>

### `estimation.inference.infer_lift_abs_se_finite`

**Cause and message:**

<p>abs_se must be finite, got {abs_se}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `abs_se`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.infer_lift_abs_se_positive"></span>

### `estimation.inference.infer_lift_abs_se_positive`

**Cause and message:**

<p>abs_se must be &gt; 0.0, got {abs_se}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `abs_se`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.joint_reference_invalid"></span>

### `estimation.inference.joint_reference_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/inference.py#L109).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.null_lift_log"></span>

### `estimation.inference.null_lift_log`

**Cause and message:**

<p>null_lift must be &gt; -1.0 (log scale), got {null_lift}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.one_sided_alpha_doubles"></span>

### `estimation.inference.one_sided_alpha_doubles`

**Cause and message:**

<p>one-sided alpha={alpha} doubles to alpha_eff={alpha_eff} &gt;= 1: no displayable two-sided interval exists at that level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `alpha_eff`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.encouragement`, `increment.estimation.inference`

<span id="estimation.inference.prior_excludes_cluster_robust_t"></span>

### `estimation.inference.prior_excludes_cluster_robust_t`

**Cause and message:**

<p>prior and a cluster-robust t or joint reference are mutually exclusive: the available covariance does not specify the requested Normal conjugate update</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.prior_excludes_joint_reference"></span>

### `estimation.inference.prior_excludes_joint_reference`

**Cause and message:**

<p>an informative prior cannot be combined with a joint frequentist relative set</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.inference`

<span id="estimation.inference.prior_excludes_welch_reference"></span>

### `estimation.inference.prior_excludes_welch_reference`

**Cause and message:**

<p>arm_ns={arm_ns!r} requests a Welch-Satterthwaite sampling reference, which cannot be combined with an informative prior: a t critical value on a prior-updated scale is neither a sampling interval nor a posterior interval</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_ns`

**Registering modules:** `increment.estimation._fixed_horizon`, `increment.estimation.inference`

<span id="estimation.inference.unknown_value_scale"></span>

### `estimation.inference.unknown_value_scale`

**Cause and message:**

<p>Unknown value_scale={value_scale!r}: must be &quot;relative&quot; or &quot;absolute&quot;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_scale`

**Registering modules:** `increment.estimation.inference`

<span id="estimation.meta.alpha_strictly_between"></span>

### `estimation.meta.alpha_strictly_between`

**Cause and message:**

<p>alpha must be strictly between 0 and 1, got {alpha}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.alpha_too_small"></span>

### `estimation.meta.alpha_too_small`

**Cause and message:**

<p>alpha is too small: alpha / 2 underflows floating-point precision</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.estimates`, `increment.estimation.meta`, `increment.estimation.quantile`

<span id="estimation.meta.est_a_est"></span>

### `estimation.meta.est_a_est`

**Cause and message:**

<p>est_a and est_b must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.est_contains_non"></span>

### `estimation.meta.est_contains_non`

**Cause and message:**

<p>est contains non-finite values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.est_var_same"></span>

### `estimation.meta.est_var_same`

**Cause and message:**

<p>est and var must have the same shape, got {est_arr} vs {var_arr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `est_arr`, `var_arr`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.est_var_shape"></span>

### `estimation.meta.est_var_shape`

**Cause and message:**

<p>est and var must be 1-D, got shape {est_arr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `est_arr`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.hksj_pooled_variance"></span>

### `estimation.meta.hksj_pooled_variance`

**Cause and message:**

<p>HKSJ pooled variance is degenerate (segment estimates are identical or numerically indistinguishable given their weights): the resulting interval would have zero width and is not meaningful; check for duplicated segments</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.marginalized_segment.finite"></span>

### `estimation.meta.marginalized_segment.finite`

**Cause and message:**

<p>{name} must be finite, got {arr!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arr`, `name`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.marginalized_segment.lb_exceed_ub"></span>

### `estimation.meta.marginalized_segment.lb_exceed_ub`

**Cause and message:**

<p>lb must not exceed ub for every segment, got lb={lb!r} ub={ub!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lb`, `ub`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.marginalized_segment.length_shape"></span>

### `estimation.meta.marginalized_segment.length_shape`

**Cause and message:**

<p>{name} must be 1-D with length k={k}, got shape {arr_shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arr_shape`, `k`, `name`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.marginalized_segment.level_contradicts_alpha"></span>

### `estimation.meta.marginalized_segment.level_contradicts_alpha`

**Cause and message:**

<p>level={level} contradicts alpha={alpha}; expected level={expected_level}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `expected_level`, `level`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.need_least_estimable"></span>

### `estimation.meta.need_least_estimable`

**Cause and message:**

<p>Need at least 2 estimable segments, got {k}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.posterior_integration_unresolved"></span>

### `estimation.meta.posterior_integration_unresolved`

**Cause and message:**

<p>the tau posterior (k={k}, tau_prior_scale={tau_prior_scale:.6g}) could not be integrated to the required accuracy (support tau in [{support[0]:.6g}, {support[1]:.6g}], estimated error {integration_error:.3g} times the tolerance, budget {node_budget} log-density evaluations); refusing rather than reporting an unconverged posterior</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `integration_error`, `k`, `node_budget`, `support`, `tau_prior_scale`

**Registering modules:** `increment.breakout.heterogeneity`, `increment.estimation.meta`

<span id="estimation.meta.tau2_finite"></span>

### `estimation.meta.tau2_finite`

**Cause and message:**

<p>tau2 must be finite and &gt;= 0, got {tau2}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tau2`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.tau_prior_scale"></span>

### `estimation.meta.tau_prior_scale`

**Cause and message:**

<p>tau_prior_scale must be finite and &gt; 0, got {tau_prior_scale}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tau_prior_scale`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.var_a_finite"></span>

### `estimation.meta.var_a_finite`

**Cause and message:**

<p>var_a must be finite and &gt; 0, got {var_a}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `var_a`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.var_b_finite"></span>

### `estimation.meta.var_b_finite`

**Cause and message:**

<p>var_b must be finite and &gt; 0, got {var_b}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `var_b`

**Registering modules:** `increment.estimation.meta`

<span id="estimation.meta.var_finite_strictly"></span>

### `estimation.meta.var_finite_strictly`

**Cause and message:**

<p>var must be finite and strictly positive for every segment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.meta`, `increment.power.core`

<span id="estimation.posterior.invalid_request"></span>

### `estimation.posterior.invalid_request`

**Cause and message:**

<p>Invalid posterior {operation}: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `operation`, `reason`

**Registering modules:** `increment.estimation._relative_posterior`

<span id="estimation.posterior.numerical_unavailable"></span>

### `estimation.posterior.numerical_unavailable`

**Cause and message:**

<p>Posterior {operation} is numerically unavailable: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `_PosteriorNumericalError`

**Context keys:** `operation`, `reason`

**Registering modules:** `increment.estimation._relative_posterior`

<span id="estimation.priors.mixture.every_sigma"></span>

### `estimation.priors.mixture.every_sigma`

**Cause and message:**

<p>every mixture sigma must be &gt; 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture.every_weight"></span>

### `estimation.priors.mixture.every_weight`

**Cause and message:**

<p>every mixture weight must be &gt; 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture.every_weight_mean"></span>

### `estimation.priors.mixture.every_weight_mean`

**Cause and message:**

<p>every mixture weight, mean, and sigma must be finite, got weights={weights}, means={means}, sigmas={sigmas}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `means`, `sigmas`, `weights`

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture.prior_needs_least"></span>

### `estimation.priors.mixture.prior_needs_least`

**Cause and message:**

<p>a mixture prior needs at least one component</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture.weights_means_sigmas"></span>

### `estimation.priors.mixture.weights_means_sigmas`

**Cause and message:**

<p>weights/means/sigmas must have the same length, got {k}/{means}/{sigmas}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k`, `means`, `sigmas`

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture.weights_sum"></span>

### `estimation.priors.mixture.weights_sum`

**Cause and message:**

<p>mixture weights must sum to 1, got {total}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `total`

**Registering modules:** `increment.semantics.models`

<span id="estimation.priors.mixture_posterior.expected_negative_part_scale"></span>

### `estimation.priors.mixture_posterior.expected_negative_part_scale`

**Cause and message:**

<p>scale must be &#x27;linear&#x27; or &#x27;log&#x27;, got {scale!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `scale`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.priors.mixture_posterior.expected_positive_part_scale"></span>

### `estimation.priors.mixture_posterior.expected_positive_part_scale`

**Cause and message:**

<p>scale must be &#x27;linear&#x27; or &#x27;log&#x27;, got {scale!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `scale`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.priors.mixture_posterior.probability"></span>

### `estimation.priors.mixture_posterior.probability`

**Cause and message:**

<p>probability must be in (0, 1), got {probability}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `probability`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.priors.mixture_posterior.tail_probability"></span>

### `estimation.priors.mixture_posterior.tail_probability`

**Cause and message:**

<p>tail_probability must be in (0, 1), got {tail_probability}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tail_probability`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.priors.standard_error"></span>

### `estimation.priors.standard_error`

**Cause and message:**

<p>standard_error must be &gt; 0, got {standard_error}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `standard_error`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.priors.student_t.nu_exceeds_200"></span>

### `estimation.priors.student_t.nu_exceeds_200`

**Cause and message:**

<p>nu={nu} exceeds 200, where the mixture expansion is numerically unstable; and a t with nu &gt; 200 is within 0.6% of a Normal anyway. Use Normal(mu=0, sigma=scale) instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `nu`

**Registering modules:** `increment.estimation.priors`

<span id="estimation.quantile.control_group_found"></span>

### `estimation.quantile.control_group_found`

**Cause and message:**

<p>control_group {control_group!r} not found; groups: {arms}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `control_group`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.degenerate_spread_order"></span>

### `estimation.quantile.degenerate_spread_order`

**Cause and message:**

<p>degenerate spread at q={q}: order-statistic bounds coincide (all values in the bracket identical)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `q`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.estimate_quantile.metric_arm"></span>

### `estimation.quantile.estimate_quantile.metric_arm`

**Cause and message:**

<p>quantile metric {metric!r}, arm {arm!r}: {error}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `error`, `metric`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.log_quantile_se"></span>

### `estimation.quantile.log_quantile_se`

**Cause and message:**

<p>q must be in (0, 1), got {q}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `q`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.quantile_positive_log"></span>

### `estimation.quantile.quantile_positive_log`

**Cause and message:**

<p>q={q} quantile must be positive for the log-scale lift transport (point={point:.6g}, lower order stat={lo:.6g}); collect more units, target a higher quantile, or narrow the enrolled population until this quantile clears zero</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lo`, `point`, `q`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.too_small_bound"></span>

### `estimation.quantile.too_small_bound`

**Cause and message:**

<p>n={n} is too small to bound the q={q} quantile at level {level:.2f}; the distribution-free order-statistic bound needs n &gt;= {n_min} per arm; collect more units or target a less extreme quantile</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `level`, `n`, `n_min`, `q`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.values_contain_non"></span>

### `estimation.quantile.values_contain_non`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/quantile.py#L127).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_bad`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.quantile.values_one_dimensional"></span>

### `estimation.quantile.values_one_dimensional`

**Cause and message:**

<p>values must be one-dimensional, got shape {shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `shape`

**Registering modules:** `increment.estimation.quantile`

<span id="estimation.results.binomial.counts_out_of_range"></span>

### `estimation.results.binomial.counts_out_of_range`

**Cause and message:**

<p>binomial confidence set counts out of range: x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_c`, `n_t`, `x_c`, `x_t`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.binomial.decision_metadata"></span>

### `estimation.results.binomial.decision_metadata`

**Cause and message:**

<p>binomial confidence set has inconsistent decision metadata: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.binomial.interval_inverted"></span>

### `estimation.results.binomial.interval_inverted`

**Cause and message:**

<p>binomial confidence set has lower={lower} &gt; upper={upper}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.binomial.obsolete_construction"></span>

### `estimation.results.binomial.obsolete_construction`

**Cause and message:**

<p>binomial confidence set records method {method!r} (None: none recorded); this version reads only {supported!r}. A set naming an incompatible construction, or persisted without naming one, may use different envelopes, thresholds or numerical rules, so its stored interval and a recomputed verdict could disagree. Re-run the analysis to cut the set again.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `method`, `supported`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.binomial.posterior_unavailable"></span>

### `estimation.results.binomial.posterior_unavailable`

**Cause and message:**

<p>metric={metric!r} group_id={group_id!r}: no Normal/lognormal posterior exists for a reference_kind=&#x27;binomial&#x27; row; the exact binomial method is a frequentist test-inversion, not a posterior; chance_to_beat/prob_beyond/prob_within/risk_if_shipped and their favorable variants are unavailable here. Use stat_sig()/p_value() (both binomial-set-aware) or the persisted lift/binomial_set bounds directly.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.delta.counts_out_of_range"></span>

### `estimation.results.delta.counts_out_of_range`

**Cause and message:**

<p>delta-route conversion evidence counts out of range: x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_c`, `n_t`, `x_c`, `x_t`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.delta.dense_level"></span>

### `estimation.results.delta.dense_level`

**Cause and message:**

<p>delta-route conversion evidence is inconsistent: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.delta.reference_mismatch"></span>

### `estimation.results.delta.reference_mismatch`

**Cause and message:**

<p>estimate row for {metric!r}/{group_id!r}: delta-route conversion evidence belongs only to a reference_kind=&#x27;t&#x27; row on the log scale with no binomial_set, got reference_kind={reference_kind!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reference_kind`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.alpha_none_lb"></span>

### `estimation.results.estimate.alpha_none_lb`

**Cause and message:**

<p>alpha must be None when lb/ub are None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.interval_inverted_lb"></span>

### `estimation.results.estimate.interval_inverted_lb`

**Cause and message:**

<p>interval is inverted: lb={lb} &gt; ub={ub}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lb`, `ub`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.lb_ub_both"></span>

### `estimation.results.estimate.lb_ub_both`

**Cause and message:**

<p>lb and ub must both be set, or both be None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.level_alpha_because"></span>

### `estimation.results.estimate.level_alpha_because`

**Cause and message:**

<p>level=1.0 requires alpha because the effective tail is unrepresentable from the rounded confidence level</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.level_contradicts_effective"></span>

### `estimation.results.estimate.level_contradicts_effective`

**Cause and message:**

<p>level={level} contradicts effective alpha={alpha}; expected level={expected_level}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `expected_level`, `level`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.level_interval"></span>

### `estimation.results.estimate.level_interval`

**Cause and message:**

<p>level must be in (0, 1] for an interval, got {level}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `level`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.level_lb_ub"></span>

### `estimation.results.estimate.level_lb_ub`

**Cause and message:**

<p>level is required when lb/ub are set</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.level_none_lb"></span>

### `estimation.results.estimate.level_none_lb`

**Cause and message:**

<p>level must be None when lb/ub are None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.open_side_bound_must_be_none"></span>

### `estimation.results.estimate.open_side_bound_must_be_none`

**Cause and message:**

<p>open_side is set; {field} must be None (never a numeric sentinel), got a finite value</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.estimate.open_side_needs_calibrated_bound"></span>

### `estimation.results.estimate.open_side_needs_calibrated_bound`

**Cause and message:**

<p>open_side is set but the calibrated bound on that side is None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.joint.invalid_reference"></span>

### `estimation.results.joint.invalid_reference`

**Cause and message:**

<p>Invalid joint relative reference: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.joint.invalid_set"></span>

### `estimation.results.joint.invalid_set`

**Cause and message:**

<p>Invalid relative confidence set: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.joint.unavailable"></span>

### `estimation.results.joint.unavailable`

**Cause and message:**

<p>Joint relative inference is unavailable: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.absolute_alpha_mismatch"></span>

### `estimation.results.lift.absolute_alpha_mismatch`

**Cause and message:**

<p>abs_alpha={abs_alpha!r} contradicts the alpha {alpha!r} of the relative interval the same call cut</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `abs_alpha`, `alpha`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.absolute_alpha_without_interval"></span>

### `estimation.results.lift.absolute_alpha_without_interval`

**Cause and message:**

<p>abs_alpha={abs_alpha!r} is the alpha an additive interval was cut at, but this row carries no additive interval (abs_lb/abs_ub)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `abs_alpha`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.absolute_reference_mismatch"></span>

### `estimation.results.lift.absolute_reference_mismatch`

**Cause and message:**

<p>absolute reference {kind!r} with df {df!r} does not fit this row: a t reference carries its df, and a score reference belongs only to an exact binomial row</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `df`, `kind`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.binomial_lift_availability"></span>

### `estimation.results.lift.binomial_lift_availability`

**Cause and message:**

<p>metric={metric!r} group_id={group_id!r}: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.dof_reference_df_mismatch"></span>

### `estimation.results.lift.dof_reference_df_mismatch`

**Cause and message:**

<p>estimate row for {metric!r}/{group_id!r}: dof={dof!r} does not match reference_df={reference_df!r}; a t-reference row&#x27;s legacy dof column must agree with the reference it was actually cut from</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dof`, `group_id`, `metric`, `reference_df`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.fcr_prior_unsupported"></span>

### `estimation.results.lift.fcr_prior_unsupported`

**Cause and message:**

<p>This legacy prior-bound interval has no proven prior-free sampling state for FCR conversion; recompute the analysis from source data.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.inference_reference_kind_mismatch"></span>

### `estimation.results.lift.inference_reference_kind_mismatch`

**Cause and message:**

<p>estimate row for {metric!r}/{group_id!r}: inference={inference!r} and reference_kind={reference_kind!r} disagree; a sequential interval has a sequential reference and a fixed-horizon interval a normal or t one</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `inference`, `metric`, `reference_kind`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_carries_no"></span>

### `estimation.results.lift.liftestimate_carries_no`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r} carries no interval; decision stats need lb/ub/level (run inference with an alpha, which is the default)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_chance_to"></span>

### `estimation.results.lift.liftestimate_chance_to`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: chance_to_beat_favorable() requires preferred_direction to be set; it was not resolved from a Metric declaration for this estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_dof_set"></span>

### `estimation.results.lift.liftestimate_dof_set`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: reference_kind=&#x27;t&#x27; but this row carries no log-scale sufficient statistics (log_mean/log_se); p_value() via the t-reference formula is undefined for linear/absolute-scale estimates (e.g. infer_ate or clustered encouragement rows)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_interval_symmetric"></span>

### `estimation.results.lift.liftestimate_interval_symmetric`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: interval is not a symmetric Normal quantile interval on the {scale} scale; decision stats are only defined for fixed-horizon estimates produced by infer_lift/infer_ate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `scale`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_level_contradicts"></span>

### `estimation.results.lift.liftestimate_level_contradicts`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: level={e_level} contradicts effective alpha={e_alpha}; expected level={expected_level}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `e_alpha`, `e_level`, `expected_level`, `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_log_se"></span>

### `estimation.results.lift.liftestimate_log_se`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: log_se must be positive to compute a t-reference p-value, got log_se={e_log_se}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `e_log_se`, `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_prior_spec"></span>

### `estimation.results.lift.liftestimate_prior_spec`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: prior_spec is set but the raw log-scale sufficient statistics are missing; not a genuine infer_lift output</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_prob_favorable"></span>

### `estimation.results.lift.liftestimate_prob_favorable`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: prob_favorable() requires preferred_direction to be set; it was not resolved from a Metric declaration for this estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_risk_if"></span>

### `estimation.results.lift.liftestimate_risk_if`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: risk_if_shipped_favorable() requires preferred_direction to be set; it was not resolved from a Metric declaration for this estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.liftestimate_value_lb"></span>

### `estimation.results.lift.liftestimate_value_lb`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: value/lb &lt;= -1 is not representable on the log scale (value={e_value}, lb={e_lb}); not a genuine infer_lift output</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `e_lb`, `e_value`, `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.open_interval_unrecoverable"></span>

### `estimation.results.lift.open_interval_unrecoverable`

**Cause and message:**

<p>The selected interval endpoint cannot be recovered from its persisted sampling reference; recompute the analysis from source data.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.open_side_alternative_mismatch"></span>

### `estimation.results.lift.open_side_alternative_mismatch`

**Cause and message:**

<p>open_side={open_side!r} does not match alternative={alternative!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternative`, `open_side`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.p_value_cluster_robust_null_abs"></span>

### `estimation.results.lift.p_value_cluster_robust_null_abs`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/results.py#L652).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reference_df`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.p_value_missing_log_mean_or_se"></span>

### `estimation.results.lift.p_value_missing_log_mean_or_se`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: the sampling reference is unavailable because log_mean or log_se is missing; recompute the analysis from sufficient source data.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.p_value_null_abs_missing_abs_se"></span>

### `estimation.results.lift.p_value_null_abs_missing_abs_se`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: null_abs is set but abs_se is unavailable (degenerate arm); the additive decision is undefined and there is no silent fallback to the relative interval</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.p_value_null_lift_not_representable_dof"></span>

### `estimation.results.lift.p_value_null_lift_not_representable_dof`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: null_lift &lt;= -1 is not representable on the log scale, got {null_lift}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `null_lift`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.posterior_components_invalid"></span>

### `estimation.results.lift.posterior_components_invalid`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: stored posterior components are invalid: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reason`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.posterior_decision_stats_cluster_robust"></span>

### `estimation.results.lift.posterior_decision_stats_cluster_robust`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/results.py#L625).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reference_df`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.posterior_decision_stats_sequential"></span>

### `estimation.results.lift.posterior_decision_stats_sequential`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: decision stats are undefined for inference={inference!r}; a sequential confidence sequence&#x27;s radius is not a posterior standard deviation, so recovering (mu, sigma) from it would give a silently wrong probability. Only inference=&#x27;fixed&#x27; (single-look) estimates support chance_to_beat/prob_beyond/prob_within/risk_if_shipped/p_value.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `inference`, `metric`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.prob_within_threshold_positive"></span>

### `estimation.results.lift.prob_within_threshold_positive`

**Cause and message:**

<p>threshold must be &gt; 0, got {threshold}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `threshold`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.prob_within_threshold_unit_interval"></span>

### `estimation.results.lift.prob_within_threshold_unit_interval`

**Cause and message:**

<p>threshold must be in (0, 1), got {threshold}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `threshold`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.reference_df_required_for_t"></span>

### `estimation.results.lift.reference_df_required_for_t`

**Cause and message:**

<p>estimate row for {metric!r}/{group_id!r}: reference_kind=&#x27;t&#x27; requires a finite reference_df &gt; 0, got {reference_df!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reference_df`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.reference_df_set_for_normal"></span>

### `estimation.results.lift.reference_df_set_for_normal`

**Cause and message:**

<p>estimate row for {metric!r}/{group_id!r}: reference_kind={reference_kind!r} requires reference_df=None, got {reference_df!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `reference_df`, `reference_kind`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.t_posterior_sufficient_statistics"></span>

### `estimation.results.lift.t_posterior_sufficient_statistics`

**Cause and message:**

<p>LiftEstimate for {metric!r}/{group_id!r}: the posterior behind a t_{{{reference_df:g}}}-reference estimate is read from the working-scale moments its interval was cut from, and this row is missing {missing}. Its endpoints are t quantiles, so inverting them with a Normal z would report sigma inflated by the t/z ratio instead; there is no fallback.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `missing`, `reference_df`

**Registering modules:** `increment.estimation.results`

<span id="estimation.results.lift.threshold_representable_log"></span>

### `estimation.results.lift.threshold_representable_log`

**Cause and message:**

<p>threshold &lt;= -1 is not representable on the log scale, got {threshold}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `threshold`

**Registering modules:** `increment.estimation.results`

<span id="estimation.rollout.cost_threshold_finite"></span>

### `estimation.rollout.cost_threshold_finite`

**Cause and message:**

<p>cost_threshold must be finite, got {cost_threshold}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cost_threshold`

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.rollout.segment_rollout.policy_value_policy"></span>

### `estimation.rollout.segment_rollout.policy_value_policy`

**Cause and message:**

<p>policy_value, policy_value_raw and selection_bias must be set together or None together</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.rollout.segment_rollout.recommendation_match_value"></span>

### `estimation.rollout.segment_rollout.recommendation_match_value`

**Cause and message:**

<p>recommendation must match the value&#x27;s sign: &#x27;rollout&#x27; iff policy_value &gt; 0, &#x27;no_net_benefit&#x27; iff policy_value &lt;= 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.rollout.segment_rollout.recommendation_refuse_carry"></span>

### `estimation.rollout.segment_rollout.recommendation_refuse_carry`

**Cause and message:**

<p>recommendation=&#x27;refuse&#x27; must carry None value fields, and &#x27;rollout&#x27;/&#x27;no_net_benefit&#x27; must carry populated ones — they travel together</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.rollout.segment_rollout.selected_bool_length"></span>

### `estimation.rollout.segment_rollout.selected_bool_length`

**Cause and message:**

<p>selected must be 1-D bool with length k={k}, got shape {shape} dtype {dtype}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dtype`, `k`, `shape`

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.rollout.segment_rollout.selection_bias_share"></span>

### `estimation.rollout.segment_rollout.selection_bias_share`

**Cause and message:**

<p>selection_bias_share must be None exactly behind a refusal or when policy_value_raw == 0, and populated otherwise</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.rollout`

<span id="estimation.sequential_inversion.accepted_coordinate_mismatch"></span>

### `estimation.sequential_inversion.accepted_coordinate_mismatch`

**Cause and message:**

<p>accepted witness must be the inward bracket endpoint</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected`, `ratio`, `side`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.bound_coordinate_mismatch"></span>

### `estimation.sequential_inversion.bound_coordinate_mismatch`

**Cause and message:**

<p>bound must equal the outward certificate coordinate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `bound`, `expected`, `side`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.endpoint_width_exceeded"></span>

### `estimation.sequential_inversion.endpoint_width_exceeded`

**Cause and message:**

<p>finite endpoint exceeds max_width</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `bracket`, `max_width`, `side`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.incomplete_finite_endpoint"></span>

### `estimation.sequential_inversion.incomplete_finite_endpoint`

**Cause and message:**

<p>finite endpoint requires two certified witnesses</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `accepted`, `bracket`, `rejected`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_alpha"></span>

### `estimation.sequential_inversion.invalid_alpha`

**Cause and message:**

<p>alpha must lie strictly between zero and one</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_bracket_length"></span>

### `estimation.sequential_inversion.invalid_bracket_length`

**Cause and message:**

<p>endpoint bracket requires two coordinates</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `bracket`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_domain_evidence_type"></span>

### `estimation.sequential_inversion.invalid_domain_evidence_type`

**Cause and message:**

<p>domain evidence must be a certified interval, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_endpoint_type"></span>

### `estimation.sequential_inversion.invalid_endpoint_type`

**Cause and message:**

<p>{side} endpoint must have a typed certificate, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `side`, `value_type`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_evidence_type"></span>

### `estimation.sequential_inversion.invalid_evidence_type`

**Cause and message:**

<p>evidence must be a likelihood certificate, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_threshold_type"></span>

### `estimation.sequential_inversion.invalid_threshold_type`

**Cause and message:**

<p>threshold must be a certified interval, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.invalid_witness_type"></span>

### `estimation.sequential_inversion.invalid_witness_type`

**Cause and message:**

<p>{witness} endpoint witness must be a typed evaluation, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`, `witness`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.missing_endpoint_reason"></span>

### `estimation.sequential_inversion.missing_endpoint_reason`

**Cause and message:**

<p>nonfinite endpoint status requires a reason</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `status`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.negative_bernoulli_bound"></span>

### `estimation.sequential_inversion.negative_bernoulli_bound`

**Cause and message:**

<p>Bernoulli lower endpoint must be nonnegative</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.rejected_coordinate_mismatch"></span>

### `estimation.sequential_inversion.rejected_coordinate_mismatch`

**Cause and message:**

<p>rejected witness must be the outward bracket endpoint</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected`, `ratio`, `side`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.reversed_bracket"></span>

### `estimation.sequential_inversion.reversed_bracket`

**Cause and message:**

<p>endpoint bracket is reversed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.uncertified_empty_endpoint"></span>

### `estimation.sequential_inversion.uncertified_empty_endpoint`

**Cause and message:**

<p>empty endpoint requires a whole-domain rejection certificate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `domain_log_e`, `side`, `threshold`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.uncertified_nonrejection"></span>

### `estimation.sequential_inversion.uncertified_nonrejection`

**Cause and message:**

<p>accepted endpoint lacks certified nonrejection</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `evidence`, `side`, `threshold`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.uncertified_rejection"></span>

### `estimation.sequential_inversion.uncertified_rejection`

**Cause and message:**

<p>rejected endpoint lacks certified rejection</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `evidence`, `side`, `threshold`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.uncertified_reversed_bounds"></span>

### `estimation.sequential_inversion.uncertified_reversed_bounds`

**Cause and message:**

<p>reversed bounds require certified emptiness</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `upper`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.uncertified_uniform_bound"></span>

### `estimation.sequential_inversion.uncertified_uniform_bound`

**Cause and message:**

<p>uniform evidence bound does not prove nonrejection</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `log_e_upper`, `side`, `threshold`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.unknown_domain"></span>

### `estimation.sequential_inversion.unknown_domain`

**Cause and message:**

<p>unknown ratio domain</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `domain`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.unknown_endpoint_status"></span>

### `estimation.sequential_inversion.unknown_endpoint_status`

**Cause and message:**

<p>unknown endpoint status</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `status`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_inversion.witness_outside_bracket"></span>

### `estimation.sequential_inversion.witness_outside_bracket`

**Cause and message:**

<p>endpoint witness lies outside its bracket</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `bracket`, `ratio`, `witness`

**Registering modules:** `increment.estimation._sequential_inversion`

<span id="estimation.sequential_likelihood.asymmetric_matrix"></span>

### `estimation.sequential_likelihood.asymmetric_matrix`

**Cause and message:**

<p>matrix must be symmetric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `matrix`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.evidence_enclosure_mismatch"></span>

### `estimation.sequential_likelihood.evidence_enclosure_mismatch`

**Cause and message:**

<p>log_e must enclose predictive minus null supremum</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `log_e`, `required`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.inconsistent_nonfinite_logs"></span>

### `estimation.sequential_likelihood.inconsistent_nonfinite_logs`

**Cause and message:**

<p>nonfinite evidence requires a reason and absent null/e logs</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `log_e`, `log_null_sup`, `reason`, `status`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.insufficient_prior_nu"></span>

### `estimation.sequential_likelihood.insufficient_prior_nu`

**Cause and message:**

<p>prior requires nu&gt;dimension-1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`, `nu`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.invalid_certificate_reason"></span>

### `estimation.sequential_likelihood.invalid_certificate_reason`

**Cause and message:**

<p>reason must be a nonempty string</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.invalid_dimension"></span>

### `estimation.sequential_likelihood.invalid_dimension`

**Cause and message:**

<p>dimension must be a positive integer</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.invalid_log_interval"></span>

### `estimation.sequential_likelihood.invalid_log_interval`

**Cause and message:**

<p>log field {field!r} must be a certified interval, got {value_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `value_type`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.matrix_dimension_mismatch"></span>

### `estimation.sequential_likelihood.matrix_dimension_mismatch`

**Cause and message:**

<p>matrix dimensions must match the mean (dimension {dimension}): got row lengths {row_lengths}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`, `row_lengths`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.merge_dimension_mismatch"></span>

### `estimation.sequential_likelihood.merge_dimension_mismatch`

**Cause and message:**

<p>cannot merge different dimensions: {left_dimension} vs {right_dimension}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `left_dimension`, `right_dimension`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.missing_finite_logs"></span>

### `estimation.sequential_likelihood.missing_finite_logs`

**Cause and message:**

<p>finite evidence requires all three intervals</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `log_e`, `log_null_sup`, `log_predictive`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.negative_bernoulli_ratio"></span>

### `estimation.sequential_likelihood.negative_bernoulli_ratio`

**Cause and message:**

<p>Bernoulli ratio must be nonnegative</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `ratio`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.non_psd_matrix"></span>

### `estimation.sequential_likelihood.non_psd_matrix`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_sequential_likelihood.py#L59).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `matrix`, `principal_minors`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonfinite_observation"></span>

### `estimation.sequential_likelihood.nonfinite_observation`

**Cause and message:**

<p>observations must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonpositive_beta_parameter"></span>

### `estimation.sequential_likelihood.nonpositive_beta_parameter`

**Cause and message:**

<p>Beta hyperparameters must be positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `parameter`, `value`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonpositive_definite_matrix"></span>

### `estimation.sequential_likelihood.nonpositive_definite_matrix`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_sequential_likelihood.py#L51).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `matrix`, `principal_minors`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonpositive_max_error"></span>

### `estimation.sequential_likelihood.nonpositive_max_error`

**Cause and message:**

<p>max_error must be positive</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `max_error`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonpositive_prior_kappa"></span>

### `estimation.sequential_likelihood.nonpositive_prior_kappa`

**Cause and message:**

<p>prior requires kappa&gt;0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kappa`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonzero_empty_mean"></span>

### `estimation.sequential_likelihood.nonzero_empty_mean`

**Cause and message:**

<p>empty state requires canonical zero mean</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.nonzero_small_sample_scatter"></span>

### `estimation.sequential_likelihood.nonzero_small_sample_scatter`

**Cause and message:**

<p>n=0 or n=1 requires zero centered scatter</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `scatter`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.row_dimension_mismatch"></span>

### `estimation.sequential_likelihood.row_dimension_mismatch`

**Cause and message:**

<p>row dimension {row_dimension} does not match the declared dimension {dimension}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`, `row_dimension`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.scatter_rank_exceeds_count"></span>

### `estimation.sequential_likelihood.scatter_rank_exceeds_count`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_sequential_likelihood.py#L85).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `scatter`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.singular_candidate_residual"></span>

### `estimation.sequential_likelihood.singular_candidate_residual`

**Cause and message:**

<p>scalar candidate crosses a singular residual</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `candidate`, `mean`, `n`, `residual`, `scatter`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.state_prior_dimension_mismatch"></span>

### `estimation.sequential_likelihood.state_prior_dimension_mismatch`

**Cause and message:**

<p>states and priors must have the same dimension: control={control_dimension}, treatment={treatment_dimension}, prior_control={prior_control_dimension}, prior_treatment={prior_treatment_dimension}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_dimension`, `prior_control_dimension`, `prior_treatment_dimension`, `treatment_dimension`

**Registering modules:** `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.successes_exceed_count"></span>

### `estimation.sequential_likelihood.successes_exceed_count`

**Cause and message:**

<p>successes cannot exceed n</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `successes`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.unknown_alternative"></span>

### `estimation.sequential_likelihood.unknown_alternative`

**Cause and message:**

<p>unknown alternative</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternative`

**Registering modules:** `increment.estimation._sequential_inversion`, `increment.estimation._sequential_likelihood`

<span id="estimation.sequential_likelihood.unknown_certificate_status"></span>

### `estimation.sequential_likelihood.unknown_certificate_status`

**Cause and message:**

<p>unknown certificate status</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `status`

**Registering modules:** `increment.estimation._sequential_likelihood`

<span id="estimation.sitewide.alpha"></span>

### `estimation.sitewide.alpha`

**Cause and message:**

<p>alpha must be in (0, 1), got {alpha!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.cluster_arm_needs_two"></span>

### `estimation.sitewide.cluster_arm_needs_two`

**Cause and message:**

<p>metric &#x27;{metric}&#x27;: cluster contrast (&#x27;{cluster}&#x27;) needs at least 2 clusters in EVERY enrolled arm to estimate a between-cluster variance for that arm, got {arm_counts!r}; a single cluster carries no between-cluster variance contribution to estimate its own arm&#x27;s Welch-Satterthwaite component from.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm_counts`, `cluster`, `metric`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.control_carries_metric"></span>

### `estimation.sitewide.control_carries_metric`

**Cause and message:**

<p>control carries metric {control!r} but target carries {target!r}; a contrast is one metric, never two</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `target`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.control_carries_study"></span>

### `estimation.sitewide.control_carries_study`

**Cause and message:**

<p>control carries study_id {control!r} but target carries {target!r}; a contrast is one experiment, never two</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `target`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.control_target_are"></span>

### `estimation.sitewide.control_target_are`

**Cause and message:**

<p>control and target are both group {control!r}; a contrast needs two distinct arms</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.counterfactual_baseline_denominator"></span>

### `estimation.sitewide.counterfactual_baseline_denominator`

**Cause and message:**

<p>counterfactual baseline denominator must be positive, got {baseline_denominator!r} (site_total_denominator={site_total_denominator!r}, delta_den={delta_den!r}, target.n_units={n_units!r}, other arms {other_arms!r}, total enrolled denominator lift {den_contribution!r}); the ratio metric is undefined against a non-positive denominator</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `baseline_denominator`, `delta_den`, `den_contribution`, `n_units`, `other_arms`, `site_total_denominator`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.counterfactual_baseline_ratio"></span>

### `estimation.sitewide.counterfactual_baseline_ratio`

**Cause and message:**

<p>counterfactual baseline ratio must be positive, got numerator {baseline_numerator!r} over denominator {baseline_denominator!r} (site_total_numerator={site_total_numerator!r}, delta_num={delta_num!r}, target.n_units={n_units!r}, other arms {other_arms!r}, total enrolled numerator lift {num_contribution!r}); relative impact is undefined against a non-positive baseline</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `baseline_denominator`, `baseline_numerator`, `delta_num`, `n_units`, `num_contribution`, `other_arms`, `site_total_numerator`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.counterfactual_baseline_volume"></span>

### `estimation.sitewide.counterfactual_baseline_volume`

**Cause and message:**

<p>counterfactual baseline volume must be positive, got {baseline_volume!r} (site_total_volume={site_total_volume!r}, delta={delta!r}, target.n_units={n_units!r}, other arms {other_arms!r} contributing {other_contribution!r}); relative impact is undefined against a non-positive baseline</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `baseline_volume`, `delta`, `n_units`, `other_arms`, `other_contribution`, `site_total_volume`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.from_iid.metric_arm_carries_ref_den"></span>

### `estimation.sitewide.from_iid.metric_arm_carries_ref_den`

**Cause and message:**

<p>metric {target!r}: {label} arm {arm!r} carries a populated ref_den, which looks like a clustered row (per-cluster size rides the den family for a sum metric); build this contrast with SitewideContrast.from_clusters, not from_iid.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `label`, `target`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.other_arms_entry_metric_mismatch"></span>

### `estimation.sitewide.other_arms_entry_metric_mismatch`

**Cause and message:**

<p>other_arms entry {arm!r} carries metric {metric!r} but the contrast is on {target!r}; netting another metric&#x27;s lift out of the site total would corrupt the counterfactual baseline</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `metric`, `target`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.other_arms_entry_study_mismatch"></span>

### `estimation.sitewide.other_arms_entry_study_mismatch`

**Cause and message:**

<p>other_arms entry {arm!r} carries study_id {study_id!r} but the contrast is on study_id {target_study_id!r}; co-enrolling an arm from a different experiment would corrupt the counterfactual baseline</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `study_id`, `target_study_id`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.other_arms_repeats"></span>

### `estimation.sitewide.other_arms_repeats`

**Cause and message:**

<p>other_arms repeats group {arm!r} (already the control, the target arm, or an earlier other_arms entry); its lift would leave the counterfactual baseline twice</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.ship_all_denominator"></span>

### `estimation.sitewide.ship_all_denominator`

**Cause and message:**

<p>ship-to-all denominator must be positive, got {shipped_denominator!r} (baseline_denominator={baseline_denominator!r}, delta_den={delta_den!r}, n_exp={n_exp!r}); the ratio metric is undefined against a non-positive denominator</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `baseline_denominator`, `delta_den`, `n_exp`, `shipped_denominator`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.sitewide_ratio.metric_arm_declares"></span>

### `estimation.sitewide.sitewide_ratio.metric_arm_declares`

**Cause and message:**

<p>metric {target!r}: {label} arm {arm!r} declares x_role={x_role!r}, which is a cluster-grain row; {remedy}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `label`, `remedy`, `target`, `x_role`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.sitewide.sitewide_ratio.metric_arm_missing"></span>

### `estimation.sitewide.sitewide_ratio.metric_arm_missing`

**Cause and message:**

<p>metric {metric!r}: arm {arm!r} is missing the denominator moments a ratio contrast requires; build it through SitewideRatioContrast.from_iid/from_clusters, never from a sum-family arm.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `metric`

**Registering modules:** `increment.estimation.sitewide`

<span id="estimation.tails.unresolvable"></span>

### `estimation.tails.unresolvable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_tails.py#L51).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tail_or_alpha`, `value`, `what`

**Registering modules:** `increment.estimation._tails`

<span id="estimation.targeting.adjustment_dtype"></span>

### `estimation.targeting.adjustment_dtype`

**Cause and message:**

<p>{caller} cannot adjust on observational adjustment column {column!r} of dtype {dtype}: an adjustment column is numeric (int/float/bool) or categorical (string); a date, time, nested or mixed-type column carries no adjustment meaning. Derive a numeric or categorical pre-exposure column upstream</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `column`, `dtype`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_empty_arm"></span>

### `estimation.targeting.bootstrap_empty_arm`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_insufficient_clusters"></span>

### `estimation.targeting.bootstrap_insufficient_clusters`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_options"></span>

### `estimation.targeting.bootstrap_options`

**Cause and message:**

<p>bootstrap_seed must be a nonnegative integer and bootstrap_repetitions an integer &gt;= 2</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_tail_resolution"></span>

### `estimation.targeting.bootstrap_tail_resolution`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_unavailable_replicate"></span>

### `estimation.targeting.bootstrap_unavailable_replicate`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.bootstrap_zero_variance"></span>

### `estimation.targeting.bootstrap_zero_variance`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.cluster_count_bounds"></span>

### `estimation.targeting.cluster_count_bounds`

**Cause and message:**

<p>n_clusters ({n_clusters}) must not exceed n_holdout ({n_holdout})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_clusters`, `n_holdout`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.cluster_count_mismatch"></span>

### `estimation.targeting.cluster_count_mismatch`

**Cause and message:**

<p>rule n_clusters ({n_clusters}) must equal validation.n_clusters ({validation_n_clusters})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_clusters`, `validation_n_clusters`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.cluster_grain_count_required"></span>

### `estimation.targeting.cluster_grain_count_required`

**Cause and message:**

<p>cluster-grain policies require a non-null n_clusters</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.cluster_ids_shape"></span>

### `estimation.targeting.cluster_ids_shape`

**Cause and message:**

<p>cluster_ids has shape {cluster_ids}, expected {y} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster_ids`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.cluster_ids_string"></span>

### `estimation.targeting.cluster_ids_string`

**Cause and message:**

<p>cluster_ids must be string-typed for a stable split, got dtype {cluster_ids!r}; cast to str before calling</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster_ids`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.covariate_rows_expected"></span>

### `estimation.targeting.covariate_rows_expected`

**Cause and message:**

<p>covariate {cov!r} has {values} rows, expected {y}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cov`, `values`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.degenerate_cluster_variance"></span>

### `estimation.targeting.degenerate_cluster_variance`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.degenerate_rank_distribution"></span>

### `estimation.targeting.degenerate_rank_distribution`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.deployment_cluster_ids_required"></span>

### `estimation.targeting.deployment_cluster_ids_required`

**Cause and message:**

<p>Cluster deployment requires aligned cluster IDs</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._deployment`

<span id="estimation.targeting.deployment_cluster_ids_shape"></span>

### `estimation.targeting.deployment_cluster_ids_shape`

**Cause and message:**

<p>Deployment cluster IDs have shape {shape}; expected ({n},)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`, `shape`

**Registering modules:** `increment.estimation._deployment`

<span id="estimation.targeting.empty_arm"></span>

### `estimation.targeting.empty_arm`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.empty_group"></span>

### `estimation.targeting.empty_group`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.evaluation_population_invalid"></span>

### `estimation.targeting.evaluation_population_invalid`

**Cause and message:**

<p>Invalid evaluation population: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.every_fraction"></span>

### `estimation.targeting.every_fraction`

**Cause and message:**

<p>every fraction must be in [0, 1], got {f}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `f`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.fraction_share_units"></span>

### `estimation.targeting.fraction_share_units`

**Cause and message:**

<p>fraction must be in [0, 1]; the share of units the rule would treat, got {fraction}. It has no default on purpose: picking the cut after seeing the group table biases the reported policy value upward by 60-120%</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fraction`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.fractions_contains_duplicates"></span>

### `estimation.targeting.fractions_contains_duplicates`

**Cause and message:**

<p>fractions contains duplicates: {fracs}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fracs`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.fractions_non_empty"></span>

### `estimation.targeting.fractions_non_empty`

**Cause and message:**

<p>fractions must be a non-empty pre-specified grid; selecting from a grid chosen after seeing results is the bias this function exists to prevent</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.half_split_holds"></span>

### `estimation.targeting.half_split_holds`

**Cause and message:**

<p>the {name} half of the split holds {n_treated} treated and {n_control} control unit(s); an honest split needs at least 2 per arm in each half. Estimate the average effect instead of validating a CATE model</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_control`, `n_treated`, `name`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.holdout_mask_shape"></span>

### `estimation.targeting.holdout_mask_shape`

**Cause and message:**

<p>holdout_mask has shape {hold}, expected {y} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `hold`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.insufficient_arm_clusters"></span>

### `estimation.targeting.insufficient_arm_clusters`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.insufficient_clusters"></span>

### `estimation.targeting.insufficient_clusters`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.invalid_deployment_grain"></span>

### `estimation.targeting.invalid_deployment_grain`

**Cause and message:**

<p>Invalid intervention/deployment grains: {intervention_grain!r}, {deploy_grain!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `deploy_grain`, `intervention_grain`

**Registering modules:** `increment.estimation._deployment`

<span id="estimation.targeting.jackknife_unavailable_replicate"></span>

### `estimation.targeting.jackknife_unavailable_replicate`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.n_groups_least"></span>

### `estimation.targeting.n_groups_least`

**Cause and message:**

<p>n_groups must be at least 2 to compare a top group against a bottom one, got {n_groups}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_groups`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.nonfinite_statistic"></span>

### `estimation.targeting.nonfinite_statistic`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2736).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.outcome_shape"></span>

### `estimation.targeting.outcome_shape`

**Cause and message:**

<p>outcome y must be 1-d, got shape {y}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.overlap.empty_arm"></span>

### `estimation.targeting.overlap.empty_arm`

**Cause and message:**

<p>overlap trimming at min_propensity={min_propensity} kept {n_kept} of {n_total} units but emptied a treatment arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `IdentificationError`

**Context keys:** `min_propensity`, `n_kept`, `n_total`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.overlap.partial_cluster"></span>

### `estimation.targeting.overlap.partial_cluster`

**Cause and message:**

<p>Cluster deployment cannot trim only some members of a cluster for overlap</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `IdentificationError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.overlap.refuse"></span>

### `estimation.targeting.overlap.refuse`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/targeting.py#L2689).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `IdentificationError`

**Context keys:** `min_propensity`, `n_outside`, `n_total`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.policy_grain_mismatch"></span>

### `estimation.targeting.policy_grain_mismatch`

**Cause and message:**

<p>Prediction cannot change the fitted policy deployment grain</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.policy_metadata"></span>

### `estimation.targeting.policy_metadata`

**Cause and message:**

<p>Policy target, grain, budget rule and fitted cutoff must agree</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.score_state_required"></span>

### `estimation.targeting.score_state_required`

**Cause and message:**

<p>A deployable policy requires its frozen fitted scoring state</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.targeting_fraction_leaves"></span>

### `estimation.targeting.targeting_fraction_leaves`

**Cause and message:**

<p>targeting fraction={fraction!r} leaves fewer than 2 held-out units in an arm (treatment={n_treatment}, control={n_control}); increase the pre-committed fraction or supply more validation units</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fraction`, `n_control`, `n_treatment`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.training_mask_shape"></span>

### `estimation.targeting.training_mask_shape`

**Cause and message:**

<p>training_mask has shape {train}, expected {y} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `train`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.treatment_shape_expected"></span>

### `estimation.targeting.treatment_shape_expected`

**Cause and message:**

<p>treatment d has shape {d}, expected {y} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `d`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.unit_ids_contains"></span>

### `estimation.targeting.unit_ids_contains`

**Cause and message:**

<p>unit_ids contains {count} duplicate id(s), e.g. {dupes!r}; a duplicated unit can land on both sides of the honest split (or twice on one side), breaking the independence the split relies on; deduplicate before calling</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `dupes`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.unit_ids_shape"></span>

### `estimation.targeting.unit_ids_shape`

**Cause and message:**

<p>unit_ids has shape {unit_ids}, expected {y} to match y</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unit_ids`, `y`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.unit_ids_string"></span>

### `estimation.targeting.unit_ids_string`

**Cause and message:**

<p>unit_ids must be string-typed for a stable split, got dtype {unit_ids!r}; cast to str before calling (e.g. unit_frame&#x27;s own unit_id column, which is already String-typed)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unit_ids`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.targeting.unsupported_unit_deployment"></span>

### `estimation.targeting.unsupported_unit_deployment`

**Cause and message:**

<p>A declared cluster intervention requires whole-cluster deployment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation._deployment`

<span id="estimation.targeting.validation_fold_holds"></span>

### `estimation.targeting.validation_fold_holds`

**Cause and message:**

<p>validation fold {label} holds {n_treated} treated and {n_control} control unit(s); selection needs at least 2 per arm in every fold; use fewer folds or estimate the average effect instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `label`, `n_control`, `n_treated`

**Registering modules:** `increment.estimation.targeting`

<span id="estimation.variance.cluster_outcome_moments_least_two_units"></span>

### `estimation.variance.cluster_outcome_moments_least_two_units`

**Cause and message:**

<p>n must be at least 2 to compute ddof=1 cluster-outcome moments (metric {metric!r}, arm {group_id!r} has n={n})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `n`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_outcome_moments_nonpositive_size"></span>

### `estimation.variance.cluster_outcome_moments_nonpositive_size`

**Cause and message:**

<p>cluster metric {metric!r} arm {group_id!r}: nonpositive mean cluster size ({size_bar!r}); corrupt cluster moments</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `size_bar`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_robust_late"></span>

### `estimation.variance.cluster_robust_late`

**Cause and message:**

<p>cluster-robust LATE needs a cluster uptake family, but this row declares x_role={x_role!r} (x carries {carries}, not a cluster uptake total); the cluster-robust LATE first stage cannot be computed. Rebuild the summary with a declared cluster AND uptake fact on an Encouragement design.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `carries`, `x_role`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_robust_late_needs_cluster_uptake_family"></span>

### `estimation.variance.cluster_robust_late_needs_cluster_uptake_family`

**Cause and message:**

<p>cluster-robust LATE needs the cluster-grain uptake family (ref_x/cx1/cx2/cxy); rebuild the summary with a declared cluster AND uptake fact on an Encouragement design.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_robust_late_needs_cxden"></span>

### `estimation.variance.cluster_robust_late_needs_cxden`

**Cause and message:**

<p>cluster-robust LATE needs cxden (cov of cluster uptake total and cluster size); rebuild the summary through group_summary(cluster=...) / from_unit_summary(cluster=...).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_uptake_moments_least_two_units"></span>

### `estimation.variance.cluster_uptake_moments_least_two_units`

**Cause and message:**

<p>n must be at least 2 to compute ddof=1 cluster-uptake moments (metric {metric!r}, arm {group_id!r} has n={n})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `n`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.cluster_uptake_moments_needs_ref_den"></span>

### `estimation.variance.cluster_uptake_moments_needs_ref_den`

**Cause and message:**

<p>ratio-family moments require ref_den/cden1/cden2 (not None)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.mean_positive_log"></span>

### `estimation.variance.mean_positive_log`

**Cause and message:**

<p>mean must be positive for log(mean), got {mean}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.metric_group_log"></span>

### `estimation.variance.metric_group_log`

**Cause and message:**

<p>metric={metric!r} group={group_id!r}: {what} {mean:.6g} &lt;= 0; log-scale relative-lift inference is undefined (zero events, or a signed metric that needs an absolute-scale estimand)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `mean`, `metric`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.positive_se_log"></span>

### `estimation.variance.positive_se_log`

**Cause and message:**

<p>n must be positive for se_log_mean, got {n}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_moments_least_two_units"></span>

### `estimation.variance.ratio_moments_least_two_units`

**Cause and message:**

<p>n must be at least 2 to compute ddof=1 ratio moments (metric {metric!r}, arm {group_id!r} has n={n}); drop or merge arm {group_id!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `n`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_moments_needs_ref_den"></span>

### `estimation.variance.ratio_moments_needs_ref_den`

**Cause and message:**

<p>ratio-family moments require ref_den/cden1/cden2/cyden (not None)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_moments_nonpositive_denominator_mean"></span>

### `estimation.variance.ratio_moments_nonpositive_denominator_mean`

**Cause and message:**

<p>ratio metric {metric!r} arm {group_id!r}: nonpositive denominator mean ({d_bar!r}); the ratio estimand sum(num)/sum(den) is undefined or sign-ambiguous; check the denominator fact/window</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `d_bar`, `group_id`, `metric`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_residual_variance"></span>

### `estimation.variance.ratio_residual_variance`

**Cause and message:**

<p>{what}: ratio residual variance is not representable in float64 (residual={residual!r}) for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}; the quadratic form overflowed at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `den_bar`, `num_bar`, `residual`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_residual_variance_negative"></span>

### `estimation.variance.ratio_residual_variance_negative`

**Cause and message:**

<p>{what}: ratio residual variance is {residual:.6g}, negative beyond floating-point rounding -- these moments are corrupt or were not produced together</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `residual`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_se_final_not_representable"></span>

### `estimation.variance.ratio_se_final_not_representable`

**Cause and message:**

<p>{what}: ratio SE is not representable in float64 (R={r!r}, se={se!r}) for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}; the ratio or its variance has overflowed at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `den_bar`, `num_bar`, `r`, `se`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_se_representable"></span>

### `estimation.variance.ratio_se_representable`

**Cause and message:**

<p>{what}: ratio SE is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}; a positive residual variance ({clamped:.6g}) underflows to a zero standard error at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `clamped`, `den_bar`, `num_bar`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_se_sqrt_not_representable"></span>

### `estimation.variance.ratio_se_sqrt_not_representable`

**Cause and message:**

<p>{what}: ratio SE is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}; sqrt(residual/n)={root:.6g} over a denominator this small overflows the exponent range</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `den_bar`, `num_bar`, `root`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_variance.correlation_not_representable"></span>

### `estimation.variance.ratio_variance.correlation_not_representable`

**Cause and message:**

<p>var_log_r for metric={metric!r} group={group_id!r}: the implied numerator/denominator correlation has log magnitude {log_rho:.6g}, astronomically beyond the Cauchy-Schwarz bound of 1; these moments are corrupt or were not produced together</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `log_rho`, `metric`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_variance.se_not_representable"></span>

### `estimation.variance.ratio_variance.se_not_representable`

**Cause and message:**

<p>var_log_r for metric={metric!r} group={group_id!r} is {se!r} for a positive variance; the true standard error is not representable in float64</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `se`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_variance.true_se_out_of_range"></span>

### `estimation.variance.ratio_variance.true_se_out_of_range`

**Cause and message:**

<p>var_log_r for metric={metric!r} group={group_id!r}: the true standard error is outside the float64 exponent range (log SE = {log_se:.6g}); it is not representable at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `log_se`, `metric`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_variance.variance_scale_not_representable"></span>

### `estimation.variance.ratio_variance.variance_scale_not_representable`

**Cause and message:**

<p>var_log_r for metric={metric!r} group={group_id!r}: log-ratio variance is {value:.6g} of the terms&#x27; own scale (log scale {log_scale:.6g}), negative beyond their floating-point rounding (~{variance_slack:.3g}); the numerator and denominator moments are inconsistent (mis-aggregated upstream or not produced together), not merely cancelled.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `log_scale`, `metric`, `value`, `variance_slack`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.ratio_variance_term"></span>

### `estimation.variance.ratio_variance_term`

**Cause and message:**

<p>{what}: the ratio variance term is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}, var_den={var_den:.6g}; r*sqrt(var_den)={rooted:.6g} squares past the exponent range</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `den_bar`, `num_bar`, `rooted`, `var_den`, `what`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.registry.no_registered_available"></span>

### `estimation.variance.registry.no_registered_available`

**Cause and message:**

<p>No {registry_name} registered for &#x27;{key}&#x27;. Available keys: {available}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `available`, `key`, `registry_name`

**Registering modules:** `increment.declarations`

<span id="estimation.variance.se_log_mean_not_representable"></span>

### `estimation.variance.se_log_mean_not_representable`

**Cause and message:**

<p>se_log_mean(var={var:.6g}, mean={mean:.6g}, n={n:.6g}) is {se!r} for a positive variance; the true standard error is not representable in float64</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean`, `n`, `se`, `var`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.se_log_mean_out_of_range"></span>

### `estimation.variance.se_log_mean_out_of_range`

**Cause and message:**

<p>se_log_mean(var={var:.6g}, mean={mean:.6g}, n={n:.6g}): the true standard error is outside the float64 exponent range (log SE = {half:.6g}); it is not representable at this scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `half`, `mean`, `n`, `var`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.variance.var_non_negative"></span>

### `estimation.variance.var_non_negative`

**Cause and message:**

<p>var must be non-negative, got {var}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `var`

**Registering modules:** `increment.estimation.variance`

<span id="estimation.winsor.cutoff_in_zero_atom"></span>

### `estimation.winsor.cutoff_in_zero_atom`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.density_required"></span>

### `estimation.winsor.density_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.density_unresolved"></span>

### `estimation.winsor.density_unresolved`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.design_unsupported"></span>

### `estimation.winsor.design_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.endpoint_unrepresentable"></span>

### `estimation.winsor.endpoint_unrepresentable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.invalid_state"></span>

### `estimation.winsor.invalid_state`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.pilot_degenerate"></span>

### `estimation.winsor.pilot_degenerate`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.pilot_negative_outcome"></span>

### `estimation.winsor.pilot_negative_outcome`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.pool_mismatch"></span>

### `estimation.winsor.pool_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.posterior_unavailable"></span>

### `estimation.winsor.posterior_unavailable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.rank_size_unsupported"></span>

### `estimation.winsor.rank_size_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.raw_state_required"></span>

### `estimation.winsor.raw_state_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.reinversion_required"></span>

### `estimation.winsor.reinversion_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.studentization_degenerate"></span>

### `estimation.winsor.studentization_degenerate`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.support_required"></span>

### `estimation.winsor.support_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="estimation.winsor.support_violation"></span>

### `estimation.winsor.support_violation`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_winsor_errors.py#L14).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._winsor_errors`

<span id="family-evidence"></span>

## `evidence.*`

<span id="evidence.assignment.identity_unavailable"></span>

### `evidence.assignment.identity_unavailable`

**Cause and message:**

<p>{constructor} cannot supply assignment identity for {operation!r}; the source retains reduced counts only. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `operation`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.assignment.invalid"></span>

### `evidence.assignment.invalid`

**Cause and message:**

<p>Invalid assignment evidence: {reason}; details: {details!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `details`, `reason`

**Registering modules:** `increment._assignment_refusals`, `increment._evidence_refusals`

<span id="evidence.assignment.replay_mismatch"></span>

### `evidence.assignment.replay_mismatch`

**Cause and message:**

<p>Assignment replay does not match the declared ledger: {details!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `details`

**Registering modules:** `increment._assignment_refusals`, `increment._evidence_refusals`

<span id="evidence.assignment.selection_unavailable"></span>

### `evidence.assignment.selection_unavailable`

**Cause and message:**

<p>Source {constructor} cannot select assignment evidence for {requested!r}; offered: {offered!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`

**Registering modules:** `increment._assignment_refusals`, `increment._evidence_refusals`

<span id="evidence.assignment.unavailable"></span>

### `evidence.assignment.unavailable`

**Cause and message:**

<p>Source {constructor} cannot provide assignment evidence; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `route`

**Registering modules:** `increment._assignment_refusals`, `increment._evidence_refusals`

<span id="evidence.breakout_unavailable"></span>

### `evidence.breakout_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested breakout {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.cluster_unavailable"></span>

### `evidence.cluster_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested cluster evidence {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.covariate_unavailable"></span>

### `evidence.covariate_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested covariates {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.grain_unavailable"></span>

### `evidence.grain_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested grain {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.joint_grouping_unavailable"></span>

### `evidence.joint_grouping_unavailable`

**Cause and message:**

<p>{constructor} cannot supply joint grouping {requested!r} from separately declared sources {offered!r}; declare all requested properties on one source</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.operation_unavailable"></span>

### `evidence.operation_unavailable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_evidence_refusals.py#L143).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `request`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`, `increment.query._native_refusals`

<span id="evidence.provider_unavailable"></span>

### `evidence.provider_unavailable`

**Cause and message:**

<p>{constructor} must declare EvidenceCapabilities or implement EvidenceProvider before it can provide evidence.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`

**Registering modules:** `increment._evidence_refusals`, `increment._source_types`

<span id="evidence.raw_outcomes_unavailable"></span>

### `evidence.raw_outcomes_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested raw outcomes {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.request_unsupported"></span>

### `evidence.request_unsupported`

**Cause and message:**

<p>{constructor} does not support evidence request {requested!r}; supported request types: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="evidence.unit_rows_unavailable"></span>

### `evidence.unit_rows_unavailable`

**Cause and message:**

<p>{constructor} cannot supply requested unit rows {requested!r}; offered: {offered!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `constructor`, `offered`, `requested`, `route`

**Registering modules:** `increment._evidence_refusals`

<span id="family-facade"></span>

## `facade.*`

<span id="facade.analysis.bh_segmented_asof"></span>

### `facade.analysis.bh_segmented_asof`

**Cause and message:**

<p>{method}: BH multiplicity is supported for breakout families, not segmented as-of series; use bonferroni or none</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.clustered_day_axis"></span>

### `facade.analysis.clustered_day_axis`

**Cause and message:**

<p>{method}() is not supported on a clustered experiment; experiment {experiment!r} declares cluster {cluster!r}, and cluster-robust inference is total-grain only (run()/srm()).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `experiment`, `method`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.contrast_unavailable"></span>

### `facade.analysis.contrast_unavailable`

**Cause and message:**

<p>{method}() is unavailable for switchback contrast evidence.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `method`

**Registering modules:** `increment._evidence_refusals`, `increment.cate`, `increment.dashboard._data`

<span id="facade.analysis.daily_compliance_dimension_unsupported"></span>

### `facade.analysis.daily_compliance_dimension_unsupported`

**Cause and message:**

<p>{method}: dimensioned daily compliance readouts are not supported; use an unsegmented daily compliance readout or an as-of readout</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.definitions_state_requires_identification"></span>

### `facade.analysis.definitions_state_requires_identification`

**Cause and message:**

<p>definitions state requires a declared parallel identification</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.builders`

<span id="facade.analysis.encouragement_daily_late"></span>

### `facade.analysis.encouragement_daily_late`

**Cause and message:**

<p>daily lift: per-day incremental LATE is statistically meaningless under a weak daily first stage; read the as-of trend with a View.asof() view instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.exploratory_metrics_sequential"></span>

### `facade.analysis.exploratory_metrics_sequential`

**Cause and message:**

<p>{method}: exploratory_metrics {names!r} are chosen after data are visible, so no registered sequential model covers them under {inference}; read the declared metrics alone, or use a fixed-horizon plan.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`, `method`, `names`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis.exploratory_metrics_source_limited"></span>

### `facade.analysis.exploratory_metrics_source_limited`

**Cause and message:**

<p>{method}: exploratory_metrics adds metrics from the saved definitions, which this source does not carry; build the experiment with from_definitions.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `method`

**Registering modules:** `increment._exploratory`

<span id="facade.analysis.invalid_mixed_assignment_policy"></span>

### `facade.analysis.invalid_mixed_assignment_policy`

**Cause and message:**

<p>on_mixed_assignment must be &#x27;error&#x27;, &#x27;warn&#x27;, or &#x27;exclude&#x27;, got {on_mixed_assignment!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `on_mixed_assignment`

**Registering modules:** `increment.builders`

<span id="facade.analysis.invalid_population"></span>

### `facade.analysis.invalid_population`

**Cause and message:**

<p>population must be &#x27;assigned&#x27; or &#x27;triggered&#x27;; got {population!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `population`

**Registering modules:** `increment._evidence_refusals`, `increment._sitewide`, `increment._source_diagnostics`

<span id="facade.analysis.invalid_store_policy"></span>

### `facade.analysis.invalid_store_policy`

**Cause and message:**

<p>store must be &#x27;auto&#x27;, &#x27;always&#x27;, or &#x27;none&#x27;; got {store!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `store`

**Registering modules:** `increment.builders`

<span id="facade.analysis.observational_day_axis"></span>

### `facade.analysis.observational_day_axis`

**Cause and message:**

<p>{method} is not supported for an observational design: confounded day-axis contrasts are refused, not silently emitted</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.sitewide_arm_not_enrolled"></span>

### `facade.analysis.sitewide_arm_not_enrolled`

**Cause and message:**

<p>arm {arm!r} is not an enrolled non-control arm for metric {metric!r} (control_group is {control_group!r}). Enrolled non-control arms: {arms!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arm`, `arms`, `control_group`, `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_arm_required"></span>

### `facade.analysis.sitewide_arm_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_sitewide.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `arms`, `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_cluster_counts_missing"></span>

### `facade.analysis.sitewide_cluster_counts_missing`

**Cause and message:**

<p>sitewide() cluster checks need native evidence with cluster-grain assignment counts; this source&#x27;s evidence for cluster &#x27;{cluster}&#x27; has none</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_cluster_metric_type"></span>

### `facade.analysis.sitewide_cluster_metric_type`

**Cause and message:**

<p>metric &#x27;{metric}&#x27; (type &#x27;{metric_type}&#x27;) cannot be served by sitewide() with a declared cluster (&#x27;{cluster}&#x27;); it does not decompose over the cluster-grain moments the clustered sitewide collapse reads.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `metric`, `metric_type`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_control_missing"></span>

### `facade.analysis.sitewide_control_missing`

**Cause and message:**

<p>control_group {control_group!r} not found in the arms of metric {metric!r}. Available groups: {available!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `control_group`, `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_metric_unaligned"></span>

### `facade.analysis.sitewide_metric_unaligned`

**Cause and message:**

<p>sitewide() cannot align metric {metric!r} (type={metric_type!r}, window_days={window_days!r}, aggregation={aggregation!r}) with the calendar-window site total. Route: {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `aggregation`, `metric`, `metric_type`, `route`, `window_days`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_no_treatment_arm"></span>

### `facade.analysis.sitewide_no_treatment_arm`

**Cause and message:**

<p>no treatment arm is enrolled for metric {metric!r}; only control_group {control_group!r} has any units</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_group`, `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_outcomes_incomplete"></span>

### `facade.analysis.sitewide_outcomes_incomplete`

**Cause and message:**

<p>sitewide() outcome evidence does not reach the experiment calendar end; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `route`

**Registering modules:** `increment.query.native_contract`, `increment.query.native_source`, `increment.query.source`

<span id="facade.analysis.sitewide_ratio_denominator_missing"></span>

### `facade.analysis.sitewide_ratio_denominator_missing`

**Cause and message:**

<p>sitewide() ratio evidence for metric {metric!r} is missing its denominator total</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_requires_randomized"></span>

### `facade.analysis.sitewide_requires_randomized`

**Cause and message:**

<p>sitewide() requires a randomized identification design; observational and encouragement designs are unsupported. Use run() for their estimands.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_triggered_cluster_unsupported"></span>

### `facade.analysis.sitewide_triggered_cluster_unsupported`

**Cause and message:**

<p>triggered sitewide does not support cluster-randomized experiment {cluster!r}; use unit-randomized definitions.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_triggered_end_missing"></span>

### `facade.analysis.sitewide_triggered_end_missing`

**Cause and message:**

<p>triggered sitewide requires a declared experiment end; provide an end aligned to the calendar window or use assigned sitewide.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_triggered_sequential_unsupported"></span>

### `facade.analysis.sitewide_triggered_sequential_unsupported`

**Cause and message:**

<p>triggered sitewide is fixed-horizon and cannot use a registered sequential plan; use a fixed-horizon plan.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.sitewide_winsorized_metric"></span>

### `facade.analysis.sitewide_winsorized_metric`

**Cause and message:**

<p>sitewide() does not support winsorized metric {metric!r}; whole-site outcomes cannot reuse the per-unit transformed experiment metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._sitewide`

<span id="facade.analysis.source_context_design_disagrees_arm"></span>

### `facade.analysis.source_context_design_disagrees_arm`

**Cause and message:**

<p>source context design disagrees with the design passed to the builder</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.builders`

<span id="facade.analysis.switchback_only"></span>

### `facade.analysis.switchback_only`

**Cause and message:**

<p>{method}() is only available for switchback evidence.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `method`

**Registering modules:** `increment._source_diagnostics`

<span id="facade.analysis.trigger_unsupported"></span>

### `facade.analysis.trigger_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_evidence_refusals.py#L112).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `experiment`, `method`, `trigger`

**Registering modules:** `increment._breakout_readouts`, `increment._day_axis`, `increment._evidence_refusals`, `increment._execute`, `increment.readouts._whole_window`

<span id="facade.analysis.uncorrected_segments_sequential"></span>

### `facade.analysis.uncorrected_segments_sequential`

**Cause and message:**

<p>uncorrected segment rows have no fixed-horizon p-values under {inference}; read the registered breakout with a View.breakout(...) view instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`

**Registering modules:** `increment._breakout_readouts`

<span id="facade.analysis.undeclared_breakout"></span>

### `facade.analysis.undeclared_breakout`

**Cause and message:**

<p>breakout {requested!r} is not declared by this experiment; declared: {declared!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `requested`

**Registering modules:** `increment._binding`, `increment._dashboard_capture`, `increment._runnability`

<span id="facade.analysis.undeclared_metric_for_dimension"></span>

### `facade.analysis.undeclared_metric_for_dimension`

**Cause and message:**

<p>{method}: dimension={dimension!r} only supports metrics already declared on this experiment; metrics= may exclude declared metrics but not add new ones for the dimensioned path. Not declared: {undeclared!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dimension`, `method`, `undeclared`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.unknown_dimension"></span>

### `facade.analysis.unknown_dimension`

**Cause and message:**

<p>{method}: dimension {dimension!r} is not a declared breakout; declared breakout properties: {declared!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `dimension`, `method`

**Registering modules:** `increment._day_axis`

<span id="facade.analysis.unknown_experiment"></span>

### `facade.analysis.unknown_experiment`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/builders.py#L49).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `requested`

**Registering modules:** `increment.builders`

<span id="facade.analysis.unknown_metric"></span>

### `facade.analysis.unknown_metric`

**Cause and message:**

<p>metric {metric!r} is not declared on this experiment. Declared metrics: {declared!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `metric`

**Registering modules:** `increment._sitewide`, `increment.readouts._whole_window`

<span id="facade.analysis_config.covariate_binding_conflict"></span>

### `facade.analysis_config.covariate_binding_conflict`

**Cause and message:**

<p>covariate alias {covariate!r} has incompatible source binding or kind across methods</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covariate`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.cuped_linear_covariates"></span>

### `facade.analysis_config.cuped_linear_covariates`

**Cause and message:**

<p>metric {metric!r} requests cuped_multicovariate without covariates declared on the method.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.cuped_linear_metric_type"></span>

### `facade.analysis_config.cuped_linear_metric_type`

**Cause and message:**

<p>metric {metric!r} has type={metric_type!r}; multicovariate CUPED supports mean and conversion metrics only.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `metric_type`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.cuped_linear_moments"></span>

### `facade.analysis_config.cuped_linear_moments`

**Cause and message:**

<p>metric {metric!r} requests cuped_multicovariate, but these aggregate moments carry no joint categorical evidence for that request; export a categorical-only request in moments format 13, or use from_unit_summary, from_unit_panel, from_definitions, or an artifact published with covariate extensions.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.cuped_scalar_moments"></span>

### `facade.analysis_config.cuped_scalar_moments`

**Cause and message:**

<p>metric {metric!r} requests named scalar CUPED covariates, but this moments format carries no covariate-name provenance; use from_unit_summary, from_unit_panel, from_definitions, or an artifact with covariate extensions.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.duplicate_metric_name"></span>

### `facade.analysis_config.duplicate_metric_name`

**Cause and message:**

<p>{caller}: duplicate metric name(s) {duplicate_names!r} in metrics=</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `duplicate_names`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.exploratory_metric_unavailable"></span>

### `facade.analysis_config.exploratory_metric_unavailable`

**Cause and message:**

<p>{caller}: exploratory_metrics cannot add {names!r}; unknown to the loaded definitions: {unknown_names!r}; already declared by the experiment (pass them in metrics= instead): {declared_names!r}. Available to add: {available_names!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available_names`, `caller`, `declared_names`, `names`, `unknown_names`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.metric_definition_mismatch"></span>

### `facade.analysis_config.metric_definition_mismatch`

**Cause and message:**

<p>{caller}: metric object(s) {mismatched_names!r} differ from their declared definitions; select by name or pass a field-equal declared metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `mismatched_names`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.metric_no_resolved"></span>

### `facade.analysis_config.metric_no_resolved`

**Cause and message:**

<p>metric {metric_name!r} has no resolved source configuration</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_name`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.multifamily_scalar_breakout"></span>

### `facade.analysis_config.multifamily_scalar_breakout`

**Cause and message:**

<p>metric {metric!r} requests multiple distinct scalar CUPED covariate families for a breakout; one breakout moment row can retain only one x family. Run one scalar family per breakout request, or use cuped_multicovariate when the covariates form one joint family.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._analysis_config`

<span id="facade.analysis_config.unknown_metric_declared"></span>

### `facade.analysis_config.unknown_metric_declared`

**Cause and message:**

<p>{caller}: unknown metric(s) {unknown_names!r}; declared: {declared_names!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`, `declared_names`, `unknown_names`

**Registering modules:** `increment._analysis_config`

<span id="facade.compatibility_inspector.compatibility_report.overall_fields_equal"></span>

### `facade.compatibility_inspector.compatibility_report.overall_fields_equal`

**Cause and message:**

<p>overall fields must equal fixed aggregation {expected!r}, got {actual!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `actual`, `expected`

**Registering modules:** `increment._compatibility_inspector`

<span id="facade.evidence_dispatch.contrast_handler.readout_request_no"></span>

### `facade.evidence_dispatch.contrast_handler.readout_request_no`

**Cause and message:**

<p>contrast readout request has no procedure for metric {metric_name!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_name`

**Registering modules:** `increment._evidence_dispatch`

<span id="facade.plan_compatibility.plan_family.runtime_effect_limited"></span>

### `facade.plan_compatibility.plan_family.runtime_effect_limited`

**Cause and message:**

<p>runtime_effect must be limited for warnings/exclusions and not_applicable otherwise</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._plan_compatibility`

<span id="facade.study.switchback_study.allocation_include_its"></span>

### `facade.study.switchback_study.allocation_include_its`

**Cause and message:**

<p>switchback allocation must include its declared control_group</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._study`

<span id="facade.study.switchback_study.identification_allocation_exactly"></span>

### `facade.study.switchback_study.identification_allocation_exactly`

**Cause and message:**

<p>switchback identification requires allocation with exactly one treatment arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._study`

<span id="facade.study.switchback_study.identification_exactly_one"></span>

### `facade.study.switchback_study.identification_exactly_one`

**Cause and message:**

<p>switchback identification requires exactly one treatment arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._study`

<span id="facade.window.resolve_window_days"></span>

### `facade.window.resolve_window_days`

**Cause and message:**

<p>resolve_window_days for {metric_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment._window`

<span id="family-family"></span>

## `family.*`

<span id="family.evidence.incomplete"></span>

### `family.evidence.incomplete`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/family.py#L97).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `failed`, `invalid`, `missing`

**Registering modules:** `increment.estimation.family`

<span id="family-frame"></span>

## `frame.*`

<span id="frame.asof.day_axis_unorderable"></span>

### `frame.asof.day_axis_unorderable`

**Cause and message:**

<p>cannot order day-axis label(s) {labels!r} chronologically: string day-axis labels are ordered only when every distinct label is an unambiguous ISO-8601 date (&#x27;YYYY-MM-DD&#x27;, lexicographic order is chronological), every distinct label is an ISO-8601 timestamp at exactly midnight with no time zone and one spelling per calendar day (&#x27;YYYY-MM-DDT00:00:00&#x27;), or every label shares one consistent &#x27;&lt;prefix&gt;&lt;signed integer&gt;&#x27; shape (e.g. &#x27;d1&#x27;, &#x27;d-1&#x27;), with no two distinct labels parsing to the same integer. Declare an explicit date or numeric day axis instead of a free-form string label.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `labels`

**Registering modules:** `increment._day_axis`, `increment._day_axis_labels`, `increment._frame_panel`

<span id="frame.asof.quantile_unsupported"></span>

### `frame.asof.quantile_unsupported`

**Cause and message:**

<p>metric {metric!r} is a quantile metric; quantiles do not decompose into as-of moments</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`

**Registering modules:** `increment._frame_moments`, `increment.frame`

<span id="frame.duplicate_metric_name"></span>

### `frame.duplicate_metric_name`

**Cause and message:**

<p>duplicate metric name {name!r}: moments are keyed by name, so two specs sharing one name silently interleave their rows and cross-pair arms. Give each metric a distinct name.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.frame_panel.asof_moments_completed"></span>

### `frame.frame_panel.asof_moments_completed`

**Cause and message:**

<p>asof moments: completed_windows_only=True is contradictory for retention metric {name!r} with unbounded band {band!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `band`, `name`

**Registering modules:** `increment.frame`

<span id="frame.frame_panel.unit_covariate_column"></span>

### `frame.frame_panel.unit_covariate_column`

**Cause and message:**

<p>unit_frame: covariate {missing!r} is not a column in this panel frame; available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `missing`

**Registering modules:** `increment.frame`

<span id="frame.frame_panel.unit_covariate_varies"></span>

### `frame.frame_panel.unit_covariate_varies`

**Cause and message:**

<p>unit_frame: covariate {covariate!r} varies within at least one unit&#x27;s own rows ({units}); not a per-unit covariate; aggregate it to one value per unit first (e.g. take its pre-exposure value) before declaring it as an adjustment covariate.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covariate`, `units`

**Registering modules:** `increment.frame`

<span id="frame.frame_totals.control_group_absent"></span>

### `frame.frame_totals.control_group_absent`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/frame.py#L237).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `groups`

**Registering modules:** `increment.frame`

<span id="frame.frame_totals.metric_was_declared"></span>

### `frame.frame_totals.metric_was_declared`

**Cause and message:**

<p>metric {name!r} was not declared on this source</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.frame`

<span id="frame.frame_totals.moments_metric_undeclared"></span>

### `frame.frame_totals.moments_metric_undeclared`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/frame.py#L229).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `names`

**Registering modules:** `increment.frame`

<span id="frame.frame_totals.unit_covariate_column"></span>

### `frame.frame_totals.unit_covariate_column`

**Cause and message:**

<p>unit_frame: covariate column(s) {missing!r} not found in the retained frame; available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `missing`

**Registering modules:** `increment.frame`

<span id="frame.frame_totals.unit_covariate_reserved"></span>

### `frame.frame_totals.unit_covariate_reserved`

**Cause and message:**

<p>unit_frame: {column!r} is reserved for declared cluster metadata; the declared cluster is {cluster!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `column`

**Registering modules:** `increment.frame`

<span id="frame.metric.covariate_names_unique"></span>

### `frame.metric.covariate_names_unique`

**Cause and message:**

<p>metric {name!r} methods must declare unique covariate names</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.cuped_does_apply"></span>

### `frame.metric.cuped_does_apply`

**Cause and message:**

<p>metric {name!r}: CUPED does not apply to quantile metrics (no mean to adjust); remove the method CUPED declaration or use a mean-based metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.cuped_linear_supported_metrics"></span>

### `frame.metric.cuped_linear_supported_metrics`

**Cause and message:**

<p>metric {name!r}: cuped_multicovariate is supported only for mean, conversion, and ratio metrics, got {type!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `type`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.declares_cuped_linear"></span>

### `frame.metric.declares_cuped_linear`

**Cause and message:**

<p>metric {name!r} requests cuped_multicovariate without declaring covariates on that method</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.declares_cuped_method"></span>

### `frame.metric.declares_cuped_method`

**Cause and message:**

<p>metric {name!r} declares a scalar CUPED method without its required method-owned covariate.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.method_name_cuped"></span>

### `frame.metric.method_name_cuped`

**Cause and message:**

<p>metric {name!r}: Method(name=&#x27;cuped&#x27;) without variance_reduction=&#x27;cuped&#x27; would label an unadjusted estimate as CUPED-adjusted; use Method.cuped(...) or choose a different result label.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.missing_drop_renamed"></span>

### `frame.metric.missing_drop_renamed`

**Cause and message:**

<p>metric {metric!r}: missing=&#x27;drop&#x27; is now spelled missing=&#x27;skip&#x27; (the metric uses only units with a value; the units stay in every other metric). Route: declare {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.missing_impute"></span>

### `frame.metric.missing_impute`

**Cause and message:**

<p>metric {metric!r}: missing=&#x27;impute&#x27; would fill the outcome with its own mean, shrinking variance and biasing the estimate. Route: {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.scalar_and_set_covariates"></span>

### `frame.metric.scalar_and_set_covariates`

**Cause and message:**

<p>metric {name!r} cannot combine scalar and multicovariate CUPED declarations on the same method</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.sets_quantile_but"></span>

### `frame.metric.sets_quantile_but`

**Cause and message:**

<p>metric {name!r} sets quantile= but type={type!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `type`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.sets_threshold_days"></span>

### `frame.metric.sets_threshold_days`

**Cause and message:**

<p>metric {name!r} sets threshold_days but type={type!r}; the observation band belongs to type=&#x27;retention&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `type`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.type_but_sets"></span>

### `frame.metric.type_but_sets`

**Cause and message:**

<p>metric {name!r} has type={type!r} but sets numerator/denominator; those belong to type=&#x27;ratio&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `type`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.type_quantile_needs"></span>

### `frame.metric.type_quantile_needs`

**Cause and message:**

<p>metric {name!r} has type=&#x27;quantile&#x27; and needs quantile in (0, 1), got {quantile!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `quantile`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.type_ratio_but"></span>

### `frame.metric.type_ratio_but`

**Cause and message:**

<p>metric {name!r} has type=&#x27;ratio&#x27; but is missing {missing}; a ratio metric needs both a numerator and a denominator column</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`, `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.type_retention_needs"></span>

### `frame.metric.type_retention_needs`

**Cause and message:**

<p>metric {name!r}: type=&#x27;retention&#x27; needs threshold_days; an int N for the unbounded band [N, inf), or [a, b] for the half-open observation band [a, b)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.window_days_supported"></span>

### `frame.metric.window_days_supported`

**Cause and message:**

<p>metric {name!r}: window_days is not supported for quantile metrics on the frame path</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric.winsorization_applies_type"></span>

### `frame.metric.winsorization_applies_type`

**Cause and message:**

<p>metric {name!r}: winsorization applies only to type=&#x27;mean&#x27;, got {type!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `type`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric_type_quantile"></span>

### `frame.metric_type_quantile`

**Cause and message:**

<p>metric {name!r}: type=&#x27;quantile&#x27; needs quantile; use the explicit MetricSpec form: MetricSpec(name={name!r}, type=&#x27;quantile&#x27;, quantile=...)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric_type_retention"></span>

### `frame.metric_type_retention`

**Cause and message:**

<p>metric {name!r}: type=&#x27;retention&#x27; needs threshold_days; use the explicit MetricSpec form: MetricSpec(name={name!r}, type=&#x27;retention&#x27;, threshold_days=...)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metric_unknown_type"></span>

### `frame.metric_unknown_type`

**Cause and message:**

<p>metric {name!r}: unknown type {kind!r}; expected one of &#x27;mean&#x27;, &#x27;conversion&#x27;, &#x27;ratio&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `kind`, `name`

**Registering modules:** `increment._metric_specs`

<span id="frame.metrics.type_required"></span>

### `frame.metrics.type_required`

**Cause and message:**

<p>metric {metric!r} has dtype {dtype!r}; choose type=&#x27;mean&#x27; or type=&#x27;conversion&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dtype`, `metric`, `options`

**Registering modules:** `increment._metric_specs`

<span id="frame.metrics_empty_supply"></span>

### `frame.metrics_empty_supply`

**Cause and message:**

<p>metrics is empty; supply at least one metric to estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._metric_specs`

<span id="frame.metrics_entries_metricspec"></span>

### `frame.metrics_entries_metricspec`

**Cause and message:**

<p>metrics entries must be a metric name, MetricSpec, or mapping, got {type_name}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `type_name`

**Registering modules:** `increment._metric_specs`

<span id="frame.missing_policy.panel_skip"></span>

### `frame.missing_policy.panel_skip`

**Cause and message:**

<p>missing={missing!r} is not supported on the panel shape; the panel is densified to a unit x day spine with zero-filled cells, so a skipped (unit, day) row silently means zero, not an observed-only population. Declare missing=&#x27;zero&#x27; if null means &quot;no events&quot;, or collapse to one row per unit and use shape=&#x27;summary&#x27; with missing=&#x27;skip&#x27;.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `missing`

**Registering modules:** `increment._frame_validation`

<span id="frame.moments.asof_completed_requires_uptake_window"></span>

### `frame.moments.asof_completed_requires_uptake_window`

**Cause and message:**

<p>asof moments: completed_windows_only=True under an encouragement design requires bounded outcome and uptake windows</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._frame_moments`

<span id="frame.moments.asof_moments_completed"></span>

### `frame.moments.asof_moments_completed`

**Cause and message:**

<p>asof moments: completed_windows_only=True is contradictory for retention metric {metric!r} with unbounded band {band!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `band`, `metric`

**Registering modules:** `increment._frame_moments`

<span id="frame.moments.metric_type_conversion"></span>

### `frame.moments.metric_type_conversion`

**Cause and message:**

<p>metric {metric!r}: type=&#x27;conversion&#x27; collapsed to a per-unit value of {value:g} (&gt; 1); summing per-day rows yields a day count, not a 0/1 conversion flag. If you want the number of converting days, declare a mean metric with a count aggregation instead; for any-occurrence conversion, set window_days on this metric or pre-reduce the column to one 0/1 row per unit.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `value`

**Registering modules:** `increment._frame_moments`

<span id="frame.panel.declaration_canonical_collision"></span>

### `frame.panel.declaration_canonical_collision`

**Cause and message:**

<p>from_unit_panel declaration {role!r} uses canonical column name {canonical_name!r} as {source_name!r}; rename that declaration column before constructing the panel. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `canonical_name`, `role`, `route`, `source_name`

**Registering modules:** `increment._frame_panel`

<span id="frame.validation.breakout_overlaps_column"></span>

### `frame.validation.breakout_overlaps_column`

**Cause and message:**

<p>breakout {name!r} overlaps {role} column {name!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `role`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.breakout_overlaps_metric"></span>

### `frame.validation.breakout_overlaps_metric`

**Cause and message:**

<p>breakout {name!r} overlaps metric input column {name!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.breakout_reserved_internal"></span>

### `frame.validation.breakout_reserved_internal`

**Cause and message:**

<p>breakout {name!r} is reserved for internal panel/moment fields</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.breakout_unit_stable"></span>

### `frame.validation.breakout_unit_stable`

**Cause and message:**

<p>breakout {name!r} is not unit-stable: {count} unit(s) have several values, including {shown}{more}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `more`, `name`, `shown`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.covariate_not_numeric"></span>

### `frame.validation.covariate_not_numeric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L232).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covariate`, `dtype`, `metric`, `variance_reduction`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.design_control_group"></span>

### `frame.validation.design_control_group`

**Cause and message:**

<p>design.control_group={control_group!r} disagrees with control={control!r}; one control, declared once</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `control_group`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.dtype_disagree_day"></span>

### `frame.validation.dtype_disagree_day`

**Cause and message:**

<p>{ds_name!r} (dtype {ds_dtype}) and {anchor_name!r} (dtype {anchor_dtype}) disagree on day-axis kind; one is an explicit numeric day index and the other is a calendar date/datetime. Declare both the same kind.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `anchor_dtype`, `anchor_name`, `ds_dtype`, `ds_name`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.duplicate_breakout_column"></span>

### `frame.validation.duplicate_breakout_column`

**Cause and message:**

<p>duplicate breakout column(s): {duplicates!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `duplicates`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.from_unit_panel"></span>

### `frame.validation.from_unit_panel`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L367).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covered`, `retention`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.group_label_collision"></span>

### `frame.validation.group_label_collision`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L113).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `collision`, `group`, `route`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.group_label_reserved"></span>

### `frame.validation.group_label_reserved`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L120).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group`, `labels`, `route`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.metric_covariate_column"></span>

### `frame.validation.metric_covariate_column`

**Cause and message:**

<p>metric {metric!r}: covariate column {covariate!r} has {n_missing} null/NaN value(s) and its method declares missing=&#x27;refuse&#x27;. Choose missing=&#x27;impute&#x27; to fill with the mean of the observed values (safe when which values are missing is unrelated to their size, even if one arm lost more), missing=&#x27;zero&#x27; if null means &quot;no pre-period activity&quot;, or repair upstream with increment.estimators.impute.pooled_mean.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covariate`, `metric`, `n_missing`

**Registering modules:** `increment._frame_moments`, `increment._frame_validation`, `increment.breakout.estimates`, `increment.query.native_source`, `increment.readouts._breakout`, `increment.readouts._encouragement`, `increment.readouts._metric_rows`, `increment.readouts._passes`

<span id="frame.validation.metric_covariate_impute_all_null"></span>

### `frame.validation.metric_covariate_impute_all_null`

**Cause and message:**

<p>metric {metric!r}: covariate column {covariate!r} has no observed values to impute a pooled mean from; every row is null/NaN. Drop the covariate method or repair the column upstream.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `covariate`, `metric`

**Registering modules:** `increment._frame_moments`, `increment._frame_validation`, `increment.breakout.estimates`, `increment.query.native_source`, `increment.readouts._breakout`, `increment.readouts._encouragement`, `increment.readouts._metric_rows`, `increment.readouts._passes`

<span id="frame.validation.metric_overlaps_column"></span>

### `frame.validation.metric_overlaps_column`

**Cause and message:**

<p>column {name!r} is the {role} column, so it cannot also be a metric value; measure the metric in its own column</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `role`

**Registering modules:** `increment._frame_validation`

<span id="frame.validation.metric_value_missing"></span>

### `frame.validation.metric_value_missing`

**Cause and message:**

<p>metric {metric!r}: column {column!r} has {n_missing} null/NaN value(s). If null means &quot;no events&quot;, declare MetricSpec(missing=&#x27;zero&#x27;) or run df, n = increment.estimators.impute.zeros(df, {column!r}). If the value is undefined for some units (an average order value for a unit without orders), declare a ratio metric, MetricSpec(type=&#x27;ratio&#x27;, numerator=..., denominator=...), over every unit. If the value exists but was not observed, declare missing=&#x27;skip&#x27;: the metric then uses only units with a value, which is unbiased only when missingness is unrelated to the values.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `metric`, `n_missing`

**Registering modules:** `increment._frame_validation`

<span id="frame.winsorization.collapsed_bound"></span>

### `frame.winsorization.collapsed_bound`

**Cause and message:**

<p>metric {metric!r}: percentile winsorization resolved a collapsed bound (lower={lower!r}, upper={upper!r}) on {date!r} while still clipping {clipped} value(s); pooling more data per date or using a fixed-value cap avoids this.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `clipped`, `date`, `lower`, `metric`, `upper`

**Registering modules:** `increment._frame_moments`

<span id="family-identification"></span>

## `identification.*`

<span id="identification.encouragement.exclusion_required"></span>

### `identification.encouragement.exclusion_required`

**Cause and message:**

<p>LATE requires an ExclusionRestriction with a non-empty justification; request estimands=(&#x27;itt&#x27;, &#x27;compliance&#x27;) or declare exclusion_restriction on the encouragement design (requested estimands={estimands!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `IdentificationError`

**Context keys:** `available_estimands`, `estimands`, `required_assumption`

**Registering modules:** `increment.estimation.encouragement`

<span id="family-impute"></span>

## `impute.*`

<span id="impute.impute_column_found"></span>

### `impute.impute_column_found`

**Cause and message:**

<p>impute.{helper}: column(s) {missing!r} not found in frame; available columns: {present}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `helper`, `missing`, `present`

**Registering modules:** `increment.impute`

<span id="impute.impute_name_least"></span>

### `impute.impute_name_least`

**Cause and message:**

<p>impute.{helper}: name at least one column to repair</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `helper`

**Registering modules:** `increment.impute`

<span id="impute.impute_pooled_mean"></span>

### `impute.impute_pooled_mean`

**Cause and message:**

<p>impute.pooled_mean: column {col!r} has no observed values to compute a mean from; every row is null/NaN. Repair the column upstream or drop it.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `col`

**Registering modules:** `increment.impute`

<span id="impute.impute_pooled_mean_not_finite"></span>

### `impute.impute_pooled_mean_not_finite`

**Cause and message:**

<p>impute.pooled_mean: column {col!r} has {observed} observed value(s) whose mean is {mean!r}, not a finite number; the values are non-finite or large enough to overflow when averaged, so there is no usable fill value. Repair the column upstream.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `col`, `mean`, `observed`

**Registering modules:** `increment.impute`

<span id="impute.impute_roles_columns"></span>

### `impute.impute_roles_columns`

**Cause and message:**

<p>impute.pooled_mean: roles name column(s) {columns!r} that are not among the columns to repair {requested!r}; declare a role only for a column you pass.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `requested`

**Registering modules:** `increment.impute`

<span id="impute.impute_roles_value"></span>

### `impute.impute_roles_value`

**Cause and message:**

<p>impute.pooled_mean: roles for column(s) {columns!r} must be &#x27;covariate&#x27; or &#x27;outcome&#x27;.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment.impute`

<span id="impute.pooled_mean_outcome"></span>

### `impute.pooled_mean_outcome`

**Cause and message:**

<p>impute.pooled_mean: column(s) {columns!r} are declared outcomes, and filling an outcome with its pooled mean shrinks variance and can bias the estimate. Route: {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `route`

**Registering modules:** `increment.impute`

<span id="family-install"></span>

## `install.*`

<span id="install.dashboard_extra_missing"></span>

### `install.dashboard_extra_missing`

**Cause and message:**

<p>{feature} requires the dashboard extra. Install it with {install}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `MissingExtraError`

**Context keys:** `feature`, `install`

**Registering modules:** `increment._optional`

<span id="install.tables_extra_missing"></span>

### `install.tables_extra_missing`

**Cause and message:**

<p>{feature} requires the tables extra. Install it with {install}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `MissingExtraError`

**Context keys:** `feature`, `install`

**Registering modules:** `increment._optional`

<span id="install.warehouse_extra_missing"></span>

### `install.warehouse_extra_missing`

**Cause and message:**

<p>{feature} requires the warehouse extra. Install it with {install}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `MissingExtraError`

**Context keys:** `feature`, `install`

**Registering modules:** `increment._optional`

<span id="family-integrity"></span>

## `integrity.*`

<span id="integrity.allocation_missing_or_nonconstant"></span>

### `integrity.allocation_missing_or_nonconstant`

**Cause and message:**

<p>SRM requires a declared target allocation for population {population!r}. Declare Randomized(control_group=..., allocation={{...}}, allocation_scheme=...) and retry.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `population`

**Registering modules:** `increment.estimation.assignment_integrity`, `increment.readouts._common`

<span id="integrity.allocation_scheme_missing"></span>

### `integrity.allocation_scheme_missing`

**Cause and message:**

<p>SRM requires a declared target allocation for population {population!r}. Declare Randomized(control_group=..., allocation={{...}}, allocation_scheme=...) and retry.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `population`

**Registering modules:** `increment.estimation.assignment_integrity`, `increment.readouts._common`

<span id="family-kernel"></span>

## `kernel.*`

<span id="kernel.context.closed"></span>

### `kernel.context.closed`

**Cause and message:**

<p>Execution context is closed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.context.undeclared_stream"></span>

### `kernel.context.undeclared_stream`

**Cause and message:**

<p>Execution context does not declare stream {stream!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `stream`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.definitions.catalog_invalid_destination"></span>

### `kernel.definitions.catalog_invalid_destination`

**Cause and message:**

<p>Catalog destination {path!r} is not an empty path or an Increment catalog; choose a new catalog file outside the definitions, or use the catalog file directly</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `path`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.definitions.catalog_missing"></span>

### `kernel.definitions.catalog_missing`

**Cause and message:**

<p>Offline definitions catalog is missing for {path!r}; bind the definitions with a connection first</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `path`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.definitions.catalog_stale"></span>

### `kernel.definitions.catalog_stale`

**Cause and message:**

<p>Offline definitions catalog is stale for {path!r}; rebind the definitions with a connection</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `path`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.definitions.catalog_unwritable"></span>

### `kernel.definitions.catalog_unwritable`

**Cause and message:**

<p>Could not write the definitions catalog at {path!r}; pass catalog_path pointing to a writable file location</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `path`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.method.duplicate_instance"></span>

### `kernel.method.duplicate_instance`

**Cause and message:**

<p>Method instance id {method_instance_id!r} is duplicated</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `method_instance_id`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.method.implementation_ref_missing"></span>

### `kernel.method.implementation_ref_missing`

**Cause and message:**

<p>Method callable field {field!r} has no implementation reference</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.provider.capabilities_missing"></span>

### `kernel.provider.capabilities_missing`

**Cause and message:**

<p>The provider declares no EvidenceCapabilities; set its evidence attribute before verifying it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CodedError`

**Context keys:** —

**Registering modules:** `increment._source_types`

<span id="kernel.provider.capability_mismatch"></span>

### `kernel.provider.capability_mismatch`

**Cause and message:**

<p>Source {source_ref!r} provides different evidence than its experiment recorded; build the experiment again from the current source</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CodedError`

**Context keys:** `source_ref`

**Registering modules:** `increment._source_types`

<span id="kernel.provider.digest_mismatch"></span>

### `kernel.provider.digest_mismatch`

**Cause and message:**

<p>The declarations recorded for sources {source_refs!r} do not match the provider; pass the experiment built from this source, or build both again</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CodedError`

**Context keys:** `source_refs`

**Registering modules:** `increment._source_types`

<span id="kernel.provider.mismatch"></span>

### `kernel.provider.mismatch`

**Cause and message:**

<p>This is not the provider the BoundResolver was built with; verify that provider</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CodedError`

**Context keys:** —

**Registering modules:** `increment._source_types`

<span id="kernel.request.capture_not_supported"></span>

### `kernel.request.capture_not_supported`

**Cause and message:**

<p>Saved capture is an execution step; call increment.workflows.capture_sequential(experiment, reader, ...) then run</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.request.datetime_timezone"></span>

### `kernel.request.datetime_timezone`

**Cause and message:**

<p>Request datetime field {field!r} must be timezone-aware</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.request.invalid"></span>

### `kernel.request.invalid`

**Cause and message:**

<p>Invalid analysis request: {field}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.request.population_scope_only"></span>

### `kernel.request.population_scope_only`

**Cause and message:**

<p>Set population on the view, as in View.overall(population=...)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.request.scope_mismatch"></span>

### `kernel.request.scope_mismatch`

**Cause and message:**

<p>Request kind {kind!r} field {field!r} disagrees with its view</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `kind`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.request.unportable"></span>

### `kernel.request.unportable`

**Cause and message:**

<p>Request kind {kind!r} field {field!r} is not portable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `kind`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.run.reader_closed"></span>

### `kernel.run.reader_closed`

**Cause and message:**

<p>The reader for experiment {experiment_id!r} is closed; build a new reader with the builder that returned it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment_id`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.run.reader_offline"></span>

### `kernel.run.reader_offline`

**Cause and message:**

<p>The reader for experiment {experiment_id!r} was read from an offline catalog and cannot read data; build it with a connection to run</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment_id`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.run.reader_required"></span>

### `kernel.run.reader_required`

**Cause and message:**

<p>Pass the reader a builder returns with the experiment, not a {received}; a BoundResolver only verifies a custom provider, so build a reader with from_frame, from_definitions or another builder to run</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `received`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.run.reader_unportable"></span>

### `kernel.run.reader_unportable`

**Cause and message:**

<p>The reader for experiment {experiment_id!r} holds the data or connection it reads, so it cannot be pickled or copied; pickle the experiment and build a new reader where it runs</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment_id`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.scope.family_mismatch"></span>

### `kernel.scope.family_mismatch`

**Cause and message:**

<p>A switchback view requires a switchback experiment; use run, daily, asof, or breakout with this experiment.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `experiment_id`, `view`

**Registering modules:** `increment._runnability`

<span id="kernel.scope.invalid"></span>

### `kernel.scope.invalid`

**Cause and message:**

<p>Invalid view field {field!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`

**Registering modules:** `increment.kernel.contracts`

<span id="kernel.snapshot.integrity"></span>

### `kernel.snapshot.integrity`

**Cause and message:**

<p>Captured stream {stream!r} can no longer be replayed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `snapshot_id`, `stream`

**Registering modules:** `increment.kernel.contracts`

<span id="family-logged_policy"></span>

## `logged_policy.*`

<span id="logged_policy.inference.adaptive_logging_unsupported"></span>

### `logged_policy.inference.adaptive_logging_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L198).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `logging_policy_keys`, `logging_policy_versions`, `update_batches`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.inference.alpha"></span>

### `logged_policy.inference.alpha`

**Cause and message:**

<p>alpha must be a float in (0, 1), got {alpha!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.inference.cluster_floor"></span>

### `logged_policy.inference.cluster_floor`

**Cause and message:**

<p>unit-clustered sandwich inference needs at least {minimum_total_units} independent units, got {n_units} (ESS by time {ess_by_time!r}, max cumulative weight by time {max_weight_by_time!r}, min propensity by time {min_propensity_by_time!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `ess_by_time`, `max_weight_by_time`, `min_propensity_by_time`, `minimum_total_units`, `n_units`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.inference.unrepresentable"></span>

### `logged_policy.inference.unrepresentable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L217).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `estimate`, `nonfinite`, `reference_value`, `se`, `target_value`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.policy.json_context_key"></span>

### `logged_policy.policy.json_context_key`

**Cause and message:**

<p>policy {policy_id}/{version} has context-table keys of type {key_types!r} that JSON cannot represent without changing the policy; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `key_types`, `policy_id`, `route`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.registry.duplicate"></span>

### `logged_policy.registry.duplicate`

**Cause and message:**

<p>policy {policy_id}/{version} is registered twice; a version is immutable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `policy_id`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.support.action_unsupported"></span>

### `logged_policy.support.action_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L184).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `action`, `decision_index`, `policy_id`, `policy_probability`, `unit_id`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.support.context_unsupported"></span>

### `logged_policy.support.context_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L166).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `context`, `context_key`, `policy_id`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.support.ess_floor"></span>

### `logged_policy.support.ess_floor`

**Cause and message:**

<p>effective sample size for {policy_id}/{version} at decision_index={decision_index} is {ess:.4f}, below the floor {floor}; cumulative weights are concentrated on fewer than {floor} effective units (target ESS by time {target_ess_by_time!r}, reference ESS by time {reference_ess_by_time!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `decision_index`, `ess`, `floor`, `policy_id`, `reference_ess_by_time`, `target_ess_by_time`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.support.policy_not_normalized"></span>

### `logged_policy.support.policy_not_normalized`

**Cause and message:**

<p>policy {policy_id}/{version} assigns {probabilities!r} over the candidate set at context {context!r}; probabilities must lie in [0, 1] and sum to 1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `context`, `policy_id`, `probabilities`, `version`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.admission_route"></span>

### `logged_policy.trace.admission_route`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L86).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternatives`, `supplied`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.boundary_awareness_mixed"></span>

### `logged_policy.trace.boundary_awareness_mixed`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L144).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `decision_time`, `reward_observation_boundary`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.boundary_before_decision"></span>

### `logged_policy.trace.boundary_before_decision`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L136).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `decision_time`, `reward_observation_boundary`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.candidate_set_mismatch"></span>

### `logged_policy.trace.candidate_set_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L111).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `candidate_actions`, `chosen_action`, `decision_index`, `expected_candidate_actions`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.context_value_unsupported"></span>

### `logged_policy.trace.context_value_unsupported`

**Cause and message:**

<p>pre-decision context contains unsupported value type {value_type!r}; use nested mappings, sequences, sets, or supported immutable scalar values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value_type`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.decision_time_awareness_mixed"></span>

### `logged_policy.trace.decision_time_awareness_mixed`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L153).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `decision_time`, `reference_decision_index`, `reference_decision_time`, `reference_unit_id`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.empty"></span>

### `logged_policy.trace.empty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L128).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructors`, `n_records`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.incomplete_horizon"></span>

### `logged_policy.trace.incomplete_horizon`

**Cause and message:**

<p>unit {unit_id!r} has {observed_horizon} closed decisions but the fixed horizon is {horizon}; every admitted unit needs all {horizon} rewards closed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `horizon`, `observed_horizon`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.logging_distribution_keys"></span>

### `logged_policy.trace.logging_distribution_keys`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L120).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `actions`, `decision_index`, `expected_actions`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.logging_distribution_misaligned"></span>

### `logged_policy.trace.logging_distribution_misaligned`

**Cause and message:**

<p>{n_records} records but {n_distributions} supplied logging distributions; build a trace with LoggedTrace.from_records or from_frame</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_distributions`, `n_records`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.logging_policy_unregistered"></span>

### `logged_policy.trace.logging_policy_unregistered`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L69).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `logging_policy_id`, `logging_policy_version`, `registered`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.missing_columns"></span>

### `logged_policy.trace.missing_columns`

**Cause and message:**

<p>trace frame is missing required columns {missing!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.open_reward"></span>

### `logged_policy.trace.open_reward`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L94).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `reward_observation_boundary`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.propensity_below_floor"></span>

### `logged_policy.trace.propensity_below_floor`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L50).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `action`, `context`, `decision_index`, `floor`, `logging_policy_id`, `logging_policy_version`, `propensity`, `route`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.propensity_missing"></span>

### `logged_policy.trace.propensity_missing`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L27).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `logging_policy_id`, `logging_policy_version`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.propensity_nonfinite"></span>

### `logged_policy.trace.propensity_nonfinite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L36).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `propensity`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.propensity_out_of_range"></span>

### `logged_policy.trace.propensity_out_of_range`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L43).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `propensity`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.propensity_stale"></span>

### `logged_policy.trace.propensity_stale`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L60).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `chosen_action`, `decision_index`, `logged`, `logging_policy_id`, `logging_policy_version`, `registered`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.registry_audit_required"></span>

### `logged_policy.trace.registry_audit_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L78).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructors`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.reward_nonfinite"></span>

### `logged_policy.trace.reward_nonfinite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/logged_policy/_refusals.py#L102).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_index`, `reward`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.unit_id_missing"></span>

### `logged_policy.trace.unit_id_missing`

**Cause and message:**

<p>trace frame row {row} has no unit_id ({unit_id!r}); every decision must belong to an identifiable unit</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `row`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="logged_policy.trace.unordered"></span>

### `logged_policy.trace.unordered`

**Cause and message:**

<p>unit {unit_id!r} decisions are not a consecutive 1..T sequence with strictly increasing decision_time: indices={decision_indices!r}, times={decision_times!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `decision_indices`, `decision_times`, `unit_id`

**Registering modules:** `increment.logged_policy`, `increment.logged_policy._refusals`

<span id="family-model"></span>

## `model.*`

<span id="model.field.length"></span>

### `model.field.length`

**Cause and message:**

<p>{model}.{field} value {value!r} violates its declared length: {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.literal"></span>

### `model.field.literal`

**Cause and message:**

<p>{model}.{field} value {value!r} is not one of {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.missing"></span>

### `model.field.missing`

**Cause and message:**

<p>{model}.{field} is required and was not supplied</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `model`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.nonfinite"></span>

### `model.field.nonfinite`

**Cause and message:**

<p>{model}.{field} value {value!r} must be finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.pattern"></span>

### `model.field.pattern`

**Cause and message:**

<p>{model}.{field} value {value!r} does not match {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.range"></span>

### `model.field.range`

**Cause and message:**

<p>{model}.{field} value {value!r} violates its declared range: {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.type"></span>

### `model.field.type`

**Cause and message:**

<p>{model}.{field} expected {constraint}, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `model`, `value`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="model.field.unknown"></span>

### `model.field.unknown`

**Cause and message:**

<p>{model} does not accept an unknown field {field!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `model`

**Registering modules:** `increment.errors`, `increment.semantics.models`, `increment.winsor`

<span id="family-moments"></span>

## `moments.*`

<span id="moments.count_field_missing"></span>

### `moments.count_field_missing`

**Cause and message:**

<p>current moments rows require {field!r}; re-export from the original data</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `field`

**Registering modules:** `increment.sources`

<span id="moments.count_not_integer"></span>

### `moments.count_not_integer`

**Cause and message:**

<p>moments field {field!r} must retain an integer, not {value!r}; re-export from the original data</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `field`, `value`

**Registering modules:** `increment.sources`

<span id="moments.count_out_of_range"></span>

### `moments.count_out_of_range`

**Cause and message:**

<p>moments field {field!r} has invalid count {value!r} for n={n}; require n &gt;= 1 and 0 &lt;= successes &lt;= n when successes is present</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `field`, `n`, `value`

**Registering modules:** `increment.sources`

<span id="moments.cuped_evidence.duplicate"></span>

### `moments.cuped_evidence.duplicate`

**Cause and message:**

<p>moments cube carries more than one CUPED evidence envelope for metric {metric!r}, population {population!r} and method {method}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `method`, `metric`, `population`

**Registering modules:** `increment._cuped_evidence`

<span id="moments.cuped_evidence.invalid"></span>

### `moments.cuped_evidence.invalid`

**Cause and message:**

<p>invalid portable CUPED evidence: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `reason`

**Registering modules:** `increment._cuped_evidence`

<span id="moments.cuped_evidence.unsupported"></span>

### `moments.cuped_evidence.unsupported`

**Cause and message:**

<p>CUPED evidence {field} {received!r} is not readable here; supported: {supported!r}. Upgrade increment, or re-export the cube from its original source.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `field`, `received`, `supported`

**Registering modules:** `increment._cuped_evidence`

<span id="moments.denominator_third_moment_missing"></span>

### `moments.denominator_third_moment_missing`

**Cause and message:**

<p>moments_format {version} denominator rows require the &#x27;cden3&#x27; field; row (metric={metric!r}, group_id={group_id!r}) omits that field. Re-export from the original data, or keep the file at the format that wrote it.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `group_id`, `metric`, `version`

**Registering modules:** `increment.sources`

<span id="moments.duplicate_key"></span>

### `moments.duplicate_key`

**Cause and message:**

<p>duplicate key {key!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `key`

**Registering modules:** `increment.sources`

<span id="moments.format.invalid"></span>

### `moments.format.invalid`

**Cause and message:**

<p>moments_format must be an exact integer version; got {received!r}; required format is {required}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `received`, `required`

**Registering modules:** `increment.sources`

<span id="moments.format.mixed"></span>

### `moments.format.mixed`

**Cause and message:**

<p>moments cube mixes wire formats {seen!r}; one cube must use format {required}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `required`, `seen`

**Registering modules:** `increment.sources`

<span id="moments.format.unsupported_future"></span>

### `moments.format.unsupported_future`

**Cause and message:**

<p>moments_format {received!r} is newer than the required format {required}. Upgrade increment to read this cube.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `received`, `required`

**Registering modules:** `increment.sources`

<span id="moments.format.unsupported_legacy"></span>

### `moments.format.unsupported_legacy`

**Cause and message:**

<p>moments_format {received!r} is a legacy format; this reader requires format {required}. Re-export the cube with workflows.export_moments().</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `received`, `required`

**Registering modules:** `increment.sources`

<span id="moments.plan.conflict"></span>

### `moments.plan.conflict`

**Cause and message:**

<p>moments cube has conflicting {field!r}: {value!r}; {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `constraint`, `field`, `value`

**Registering modules:** `increment.sources`

<span id="moments.plan.invalid"></span>

### `moments.plan.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sources.py#L306).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `available`, `constraint`, `field`, `reason`, `value`

**Registering modules:** `increment.sources`

<span id="moments.rows.duplicate"></span>

### `moments.rows.duplicate`

**Cause and message:**

<p>moments cube repeats (metric, group_id)={key!r}; one cube is one experiment&#x27;s analysis with exactly one row per arm.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `key`

**Registering modules:** `increment.sources`

<span id="moments.rows.experiment_conflict"></span>

### `moments.rows.experiment_conflict`

**Cause and message:**

<p>one moments cube cannot combine experiment identities {experiments!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `experiments`

**Registering modules:** `increment.sources`

<span id="moments.summaries.invalid"></span>

### `moments.summaries.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/summaries.py#L25).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected`, `field`, `group_id`, `metric`, `reason`, `route`, `value`

**Registering modules:** `increment.summaries`

<span id="moments.summaries.unsupported"></span>

### `moments.summaries.unsupported`

**Cause and message:**

<p>{feature} cannot be represented by per-arm mean/conversion summaries; use {route}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `feature`, `route`

**Registering modules:** `increment.summaries`

<span id="moments.v7_rows_missing_winsorization_fields"></span>

### `moments.v7_rows_missing_winsorization_fields`

**Cause and message:**

<p>moments rows are missing canonical winsorization metadata fields: {missing}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `missing`

**Registering modules:** `increment.sources`

<span id="moments.winsorization.metadata_conflict"></span>

### `moments.winsorization.metadata_conflict`

**Cause and message:**

<p>winsorization metadata for metric {metric!r} conflicts on {side!r} bound: expected percentile {expected_percentile!r}, expected bound {expected_bound!r}, received percentile {actual_percentile!r}, bound {actual_bound!r}, count {actual_n!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `actual_bound`, `actual_n`, `actual_percentile`, `expected_bound`, `expected_percentile`, `metric`, `side`

**Registering modules:** `increment.sources`

<span id="moments.winsorization.metadata_missing"></span>

### `moments.winsorization.metadata_missing`

**Cause and message:**

<p>winsorization metadata is not declared for metric {metric!r}; field {field!r} has value {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `field`, `metric`, `value`

**Registering modules:** `increment.sources`

<span id="moments.winsorization.metadata_shape"></span>

### `moments.winsorization.metadata_shape`

**Cause and message:**

<p>winsorization metadata for metric {metric!r} has invalid field {field!r} value {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `field`, `metric`, `value`

**Registering modules:** `increment.sources`

<span id="family-package"></span>

## `package.*`

<span id="package.builtin_provider_unavailable"></span>

### `package.builtin_provider_unavailable`

**Cause and message:**

<p>Installed support for {feature} is unavailable. Reinstall Increment.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CodedError`

**Context keys:** `count`, `feature`, `group`, `provider`, `reason`

**Registering modules:** `increment.errors`, `increment.kernel.ports`, `increment.query.artifact_ports`

<span id="family-plan"></span>

## `plan.*`

<span id="plan.compile_configs_template_mismatch"></span>

### `plan.compile_configs_template_mismatch`

**Cause and message:**

<p>resolved config for {metric!r} does not match metric template</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.plan`

<span id="plan.compile_configs_wrong_type"></span>

### `plan.compile_configs_wrong_type`

**Cause and message:**

<p>configs must contain ResolvedMetricConfig values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="plan.configs_contain_one"></span>

### `plan.configs_contain_one`

**Cause and message:**

<p>configs must contain one entry per metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="plan.configs_keys_match"></span>

### `plan.configs_keys_match`

**Cause and message:**

<p>configs keys must match the declared metric names exactly</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="plan.corrected_encouragement_breakout"></span>

### `plan.corrected_encouragement_breakout`

**Cause and message:**

<p>corrected encouragement breakout is not supported</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="plan.duplicate_metric_name"></span>

### `plan.duplicate_metric_name`

**Cause and message:**

<p>duplicate metric name(s) {duplicate_names!r} in metrics</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `duplicate_names`

**Registering modules:** `increment.plan`

<span id="plan.encouragement.margin"></span>

### `plan.encouragement.margin`

**Cause and message:**

<p>metric {metric!r}: margins are not supported for sequential encouragement; the sequential encouragement estimator builds no shifted null; a fixed-horizon plan applies the margin to the ITT row (valid for one planned analysis, not repeated looks)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.plan`

<span id="plan.encouragement.sequential_cuped"></span>

### `plan.encouragement.sequential_cuped`

**Cause and message:**

<p>metric {metric!r}: CUPED methods are not supported with {inference} inference for encouragement except compliance-only requests</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`, `metric`

**Registering modules:** `increment.plan`

<span id="plan.frame.override"></span>

### `plan.frame.override`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/decision.py#L194).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `metric`

**Registering modules:** `increment.decision`

<span id="plan.frame.secondaries"></span>

### `plan.frame.secondaries`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/decision.py#L209).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `derived`, `explicit`

**Registering modules:** `increment.decision`

<span id="plan.frame_contrast_does"></span>

### `plan.frame_contrast_does`

**Cause and message:**

<p>frame/contrast does not support method or prior overrides</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="plan.guardrail.direction"></span>

### `plan.guardrail.direction`

**Cause and message:**

<p>metric {metric!r}: a guardrail role requires an explicitly-set, non-neutral preferred_direction (increase|decrease) on the metric; a guardrail&#x27;s non-inferiority margin has no adverse side to test against otherwise, and &#x27;neutral&#x27; names no side</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.decision`

<span id="plan.guardrail.margin_required"></span>

### `plan.guardrail.margin_required`

**Cause and message:**

<p>metric {metric!r}: a guardrail tests that the treatment is not worse than control by more than a margin, and this guardrail declares no margin; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment.decision`

<span id="plan.metric.report_only"></span>

### `plan.metric.report_only`

**Cause and message:**

<p>metric {metric!r} is report-only (no margin/direction concept) and cannot be declared into the {role!r} testing role; it has no null it can be tested against. Report it without naming it in a plan role.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `role`

**Registering modules:** `increment.decision`

<span id="plan.metric_cuped_methods"></span>

### `plan.metric_cuped_methods`

**Cause and message:**

<p>metric {metric!r}: CUPED is unavailable under {inference} inference, including predeclared coefficients. Use raw Bernoulli outcomes without CUPED, or monitor under InferenceSpec(kind=&#x27;asymptotic_mean&#x27;) (an asymptotic confidence sequence, not the exact e-process guarantee), whose adjusted_mean and adjusted_ratio_mean laws retain the joint (Y, X) moments that construction reads (Lindon, Ham, Tingley and Bojinov 2022, arXiv:2210.08589). Pre-period coefficients are supported by its scalar_mean law through InferenceSpec.adjustments or ScalarMeanModel.adjustment</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`, `metric`

**Registering modules:** `increment.plan`

<span id="plan.metric_frame_contrast"></span>

### `plan.metric_frame_contrast`

**Cause and message:**

<p>metric {metric!r}: frame/contrast does not support method or prior configuration overrides</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.plan`

<span id="plan.metric_priors_relative"></span>

### `plan.metric_priors_relative`

**Cause and message:**

<p>metric {metric!r}: {prior} priors require the relative decision scale</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`, `prior`

**Registering modules:** `increment.plan`

<span id="plan.metrics.unknown"></span>

### `plan.metrics.unknown`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/decision.py#L183).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `path`, `unknown`

**Registering modules:** `increment.decision`

<span id="plan.observational.relative_margin"></span>

### `plan.observational.relative_margin`

**Cause and message:**

<p>relative shifted nulls are not built for the observational path: metric(s) {metrics} declare a relative margin (Metric.margin, or a plan-bound ExperimentMetric.margin); an absolute-unit margin (Metric.margin_abs) is supported</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metrics`

**Registering modules:** `increment.plan`

<span id="plan.primary.alternative_direction"></span>

### `plan.primary.alternative_direction`

**Cause and message:**

<p>metric {metric!r}: the plan tests alternative={alternative!r}, but the metric declares preferred_direction={preferred_direction!r}, so this primary could only ever show a worse result; use alternative=&#x27;two-sided&#x27; (each primary&#x27;s verdict reads its own preferred_direction) or the one-sided alternative on its favorable side</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alternative`, `metric`, `preferred_direction`

**Registering modules:** `increment.decision`

<span id="plan.unsupported_decision_plan"></span>

### `plan.unsupported_decision_plan`

**Cause and message:**

<p>unsupported decision-plan path {path!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `path`

**Registering modules:** `increment.plan`

<span id="family-power"></span>

## `power.*`

<span id="power.alpha_finite"></span>

### `power.alpha_finite`

**Cause and message:**

<p>alpha must be finite and in (0, 1) (got {alpha!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.power.sequential`

<span id="power.baseline.avg_cluster_size"></span>

### `power.baseline.avg_cluster_size`

**Cause and message:**

<p>avg_cluster_size must be &gt;= 1.0 (expected assigned units per randomized cluster), got {avg_cluster_size}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `avg_cluster_size`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_analyzed_mean"></span>

### `power.baseline.cluster_analyzed_mean`

**Cause and message:**

<p>assigned mean {avg_cluster_size}, trigger fraction {trigger_rate}, and cluster participation {cluster_participation} imply analyzed mean {analyzed_mean}; it must be finite and at least one; recount assigned units and contributing pilot clusters</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `analyzed_mean`, `avg_cluster_size`, `cluster_participation`, `route`, `trigger_rate`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_icc"></span>

### `power.baseline.cluster_icc`

**Cause and message:**

<p>cluster_icc must be in [0, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_icc_avg"></span>

### `power.baseline.cluster_icc_avg`

**Cause and message:**

<p>cluster_icc={cluster_icc} &gt; 0 requires analyzed mean cluster size &gt; 1 (got {analyzed_mean}); a one-unit contributing cluster does not identify within-cluster correlation; use a pilot with repeated observations within contributing clusters, or leave cluster_icc at 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `analyzed_mean`, `cluster_icc`, `route`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_participation"></span>

### `power.baseline.cluster_participation`

**Cause and message:**

<p>cluster_participation must be finite in (0, 1]</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster_participation`, `route`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_participation_triggered"></span>

### `power.baseline.cluster_participation_triggered`

**Cause and message:**

<p>clustered triggered planning requires cluster_participation: count contributing analyzed clusters and recruited assigned clusters in the pilot</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `avg_cluster_size`, `cluster_participation`, `route`, `trigger_rate`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_participation_untriggered"></span>

### `power.baseline.cluster_participation_untriggered`

**Cause and message:**

<p>cluster_participation must be 1.0 for an untriggered baseline</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster_participation`, `route`, `trigger_rate`

**Registering modules:** `increment.power.core`

<span id="power.baseline.cluster_size_cv"></span>

### `power.baseline.cluster_size_cv`

**Cause and message:**

<p>cluster_size_cv must be &gt;= 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.compliance"></span>

### `power.baseline.compliance`

**Cause and message:**

<p>compliance must be in (0, 1]</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.confidence"></span>

### `power.baseline.confidence`

**Cause and message:**

<p>confidence must be in (0, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.confidence_produces_non"></span>

### `power.baseline.confidence_produces_non`

**Cause and message:**

<p>confidence produces a non-finite variance inflation</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.cuped_rho"></span>

### `power.baseline.cuped_rho`

**Cause and message:**

<p>cuped_rho must be in (-1, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.icc"></span>

### `power.baseline.icc`

**Cause and message:**

<p>icc must be in [0, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.mean"></span>

### `power.baseline.mean`

**Cause and message:**

<p>mean must be &gt; 0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.mean_den"></span>

### `power.baseline.mean_den`

**Cause and message:**

<p>mean_den must be &gt; 0, got {mean_den}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean_den`

**Registering modules:** `increment.power.core`

<span id="power.baseline.summarystats_bound_pilot"></span>

### `power.baseline.summarystats_bound_pilot`

**Cause and message:**

<p>SummaryStats.n must be &gt;= 2 to bound the pilot variance&#x27;s own sampling uncertainty (got n={n}); a single unit has no degrees of freedom for a variance estimate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n`

**Registering modules:** `increment.power.core`

<span id="power.baseline.trigger_rate"></span>

### `power.baseline.trigger_rate`

**Cause and message:**

<p>trigger_rate must be in (0, 1]</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.baseline.var_zero_variance"></span>

### `power.baseline.var_zero_variance`

**Cause and message:**

<p>var must be &gt; 0; a zero-variance (deterministic) metric has no power question to answer (every solver divides by the log-ratio variance); from_proportion(p) requires 0 &lt; p &lt; 1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.binomial_arm_ceiling_below_smallest_design"></span>

### `power.binomial_arm_ceiling_below_smallest_design`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L3578).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allocation`, `conversion_inference`, `max_arm_size`, `n_c`, `n_t`

**Registering modules:** `increment.power.core`

<span id="power.binomial_arm_ceiling_exceeded"></span>

### `power.binomial_arm_ceiling_exceeded`

**Cause and message:**

<p>n_c={n_c}/n_t={n_t} analyzed units has an arm above the runtime&#x27;s ceiling of {max_arm_size} analyzed units, where it refuses every count pair, so no power can be planned; plan fewer units per arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `max_arm_size`, `n_c`, `n_t`

**Registering modules:** `increment.power.core`

<span id="power.binomial_enclosure_requires_counts"></span>

### `power.binomial_enclosure_requires_counts`

**Cause and message:**

<p>a binomial planning enclosure requires an unadjusted fixed-horizon conversion or retention count decision, not this {metric_type!r} plan; use achieved_power for model-based point power</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.power.core`

<span id="power.binomial_probability_unresolved"></span>

### `power.binomial_probability_unresolved`

**Cause and message:**

<p>Binomial planning probability is unresolved: [{lower}, {upper}] allows absolute error {error}, above {tolerance}. Use a design whose count law can be resolved within the planning bounds.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `error`, `lower`, `point`, `tolerance`, `upper`

**Registering modules:** `increment.power._binomial`, `increment.power.core`

<span id="power.binomial_replay_bound_exceeded"></span>

### `power.binomial_replay_bound_exceeded`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L2834).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cells`, `conversion_inference`, `dense_min_count`, `max_arm_size`, `max_cells`, `n_c`, `n_t`, `p_c`, `p_t`, `planned_dense_min_count`, `power`, `power_reached`

**Registering modules:** `increment.power.core`

<span id="power.binomial_size_search_unreachable"></span>

### `power.binomial_size_search_unreachable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L3480).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `conversion_inference`, `max_arm_size`, `maximum_power`, `n_per_arm`, `power`

**Registering modules:** `increment.power.core`

<span id="power.binomial_tail_level_unrepresentable"></span>

### `power.binomial_tail_level_unrepresentable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L3502).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`, `beta`, `cause`, `conversion_inference`, `decided_from`, `margin`, `n_c`, `n_t`, `p_c`, `scope`, `solver_floor`, `tail_alpha`

**Registering modules:** `increment.power.core`

<span id="power.boundary_crossing_quadrature"></span>

### `power.boundary_crossing_quadrature`

**Cause and message:**

<p>boundary-crossing quadrature did not converge: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.power.sequential`

<span id="power.bounded_metric.baseline"></span>

### `power.bounded_metric.baseline`

**Cause and message:**

<p>{metric_type} planning needs a baseline rate strictly inside (0, 1), got mean={mean!r}: a bounded metric&#x27;s control mean is a probability, and an alternative rate p0 * (1 + lift) has no room to move from a rate at or above one</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mean`, `metric_type`

**Registering modules:** `increment.power.core`

<span id="power.bounded_metric.rate_above_one"></span>

### `power.bounded_metric.rate_above_one`

**Cause and message:**

<p>{metric_type} planning at baseline rate {mean:.6g}: the {role} lift implies a rate of {implied_rate!r}, which exceeds one; an alternative that cannot occur is not a conservative approximation; lower the lift or the baseline rate</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `implied_rate`, `mean`, `metric_type`, `role`

**Registering modules:** `increment.power.core`

<span id="power.cluster.too_few_clusters"></span>

### `power.cluster.too_few_clusters`

**Cause and message:**

<p>cluster-randomized fixed-horizon power requires at least two represented clusters per arm (got k_t={k_t}, k_c={k_c}, {k_total} total)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k_c`, `k_t`, `k_total`, `recruited_k_total`, `route`

**Registering modules:** `increment.power.core`

<span id="power.core.n_per_arm_int"></span>

### `power.core.n_per_arm_int`

**Cause and message:**

<p>n_per_arm must be an integer (got {n_per_arm!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_per_arm`

**Registering modules:** `increment.power.core`

<span id="power.core.n_per_arm_min"></span>

### `power.core.n_per_arm_min`

**Cause and message:**

<p>n_per_arm must be &gt;= {minimum} for metric {metric_type!r} (got {n_per_arm})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_type`, `minimum`, `n_per_arm`

**Registering modules:** `increment.power.core`

<span id="power.cuped_linear_unavailable"></span>

### `power.cuped_linear_unavailable`

**Cause and message:**

<p>power planning does not construct the requested multicovariate CUPED estimator; plan without variance reduction, or use scalar CUPED on one continuous covariate as an approximation that is not a multicovariate CUPED power claim</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.curve.n_per_arm_int"></span>

### `power.curve.n_per_arm_int`

**Cause and message:**

<p>n_per_arm must contain integers, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.power.curve`

<span id="power.curve.n_per_arm_min"></span>

### `power.curve.n_per_arm_min`

**Cause and message:**

<p>n_per_arm must be &gt;= 2, got {value}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.power.curve`

<span id="power.empty"></span>

### `power.empty`

**Cause and message:**

<p>{name} must not be empty</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.power.curve`

<span id="power.exactly_one_relative"></span>

### `power.exactly_one_relative`

**Cause and message:**

<p>exactly one of relative_lift and target_power must be provided</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.curve`

<span id="power.finite"></span>

### `power.finite`

**Cause and message:**

<p>{name} must be finite, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `value`

**Registering modules:** `increment.power._validation`

<span id="power.fractions_nonempty"></span>

### `power.fractions_nonempty`

**Cause and message:**

<p>fractions must be nonempty</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.sequential`

<span id="power.information_fractions_lie"></span>

### `power.information_fractions_lie`

**Cause and message:**

<p>information fractions must lie in (0, 1]; got {fr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fr`

**Registering modules:** `increment.estimation.sequential`

<span id="power.information_fractions_strictly"></span>

### `power.information_fractions_strictly`

**Cause and message:**

<p>information fractions must be strictly increasing; got {fr}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fr`

**Registering modules:** `increment.estimation.sequential`

<span id="power.log_exp_needs"></span>

### `power.log_exp_needs`

**Cause and message:**

<p>log(1 - exp(x)) needs x &lt;= 0, got {x}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `x`

**Registering modules:** `increment.power.core`

<span id="power.max_workers"></span>

### `power.max_workers`

**Cause and message:**

<p>max_workers must be &gt;= 1</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.curve`

<span id="power.max_workers_positive"></span>

### `power.max_workers_positive`

**Cause and message:**

<p>max_workers must be a positive integer or None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.curve`

<span id="power.mde_relative_unavailable"></span>

### `power.mde_relative_unavailable`

**Cause and message:**

<p>mde_relative is unavailable but mde_unavailable_reason is None: a missing minimum detectable effect must state why</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.mde_unavailable_reason"></span>

### `power.mde_unavailable_reason`

**Cause and message:**

<p>mde_unavailable_reason={mde_unavailable_reason!r} contradicts an available mde_relative={mde_relative}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mde_relative`, `mde_unavailable_reason`

**Registering modules:** `increment.power.core`

<span id="power.minimum_detectable_effect.delta_method_unreliable"></span>

### `power.minimum_detectable_effect.delta_method_unreliable`

**Cause and message:**

<p>no effect that reaches power={target_power} in the {direction} direction is one the runtime can estimate: each has a planned combined log-scale standard error of at least {log_se:.4g}, and the runtime refuses a relative lift whose log-scale standard error is {log_se_limit} or more (delta_method_unreliable); raise n_per_arm</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `direction`, `log_se`, `log_se_limit`, `target_power`

**Registering modules:** `increment.power.core`

<span id="power.minimum_detectable_effect.design_search_minimum"></span>

### `power.minimum_detectable_effect.design_search_minimum`

**Cause and message:**

<p>fixed-horizon design cannot search for a minimum detectable effect when target_power={target} is at or below the null&#x27;s own crossing probability ({estimate:.6g}); zero distance already qualifies, so no strictly nonzero minimum detectable effect exists</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimate`, `target`

**Registering modules:** `increment.power.core`, `increment.power.pairwise`

<span id="power.minimum_detectable_effect.numerical_resolution"></span>

### `power.minimum_detectable_effect.numerical_resolution`

**Cause and message:**

<p>the minimum detectable effect at power={target_power} in the {direction} direction could not be resolved: distances {unresolved_interval} enclose power {power_enclosure} ({stopping_reason})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `direction`, `power_enclosure`, `stopping_reason`, `target_power`, `unresolved_interval`

**Registering modules:** `increment.power._search`, `increment.power.core`, `increment.power.sequential`

<span id="power.minimum_detectable_effect.unattainable"></span>

### `power.minimum_detectable_effect.unattainable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L2018).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `direction`, `limiting_condition`, `maximum_power`, `power_basis`, `target_power`

**Registering modules:** `increment.power.core`

<span id="power.minimum_detectable_effect.unrepresentable"></span>

### `power.minimum_detectable_effect.unrepresentable`

**Cause and message:**

<p>the minimum detectable effect at power={target_power} in the {direction} direction exists but has no float64 representation as a relative lift ({limiting_condition}); the standard error or compliance is too extreme for a relative-scale answer</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `direction`, `limiting_condition`, `target_power`

**Registering modules:** `increment.power.core`

<span id="power.need_least_segments"></span>

### `power.need_least_segments`

**Cause and message:**

<p>Need at least 2 segments for a joint Q-test power calculation, got {k}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k`

**Registering modules:** `increment.power.core`

<span id="power.nodes_multiple"></span>

### `power.nodes_multiple`

**Cause and message:**

<p>nodes must be a multiple of {panel_nodes}, got {nodes}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `nodes`, `panel_nodes`

**Registering modules:** `increment.power.sequential`

<span id="power.noncentral_t_unresolved"></span>

### `power.noncentral_t_unresolved`

**Cause and message:**

<p>noncentral-t power at noncentrality {nc!r} with {dof!r} degrees of freedom and tail allocation {tail_alpha!r} is unresolved: its tail integral did not resolve within the quadrature&#x27;s panel limits</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dof`, `nc`, `tail_alpha`

**Registering modules:** `increment.power._shared`

<span id="power.planned_looks"></span>

### `power.planned_looks`

**Cause and message:**

<p>planned_looks must be &gt;= 1, got {planned_looks}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `planned_looks`

**Registering modules:** `increment.power._validation`

<span id="power.planned_looks_positive"></span>

### `power.planned_looks_positive`

**Cause and message:**

<p>planned_looks must be a positive integer, got {planned_looks!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `planned_looks`

**Registering modules:** `increment.power._validation`

<span id="power.power.check"></span>

### `power.power.check`

**Cause and message:**

<p>power must be in [0, 1], got {power}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `power`

**Registering modules:** `increment.power.core`

<span id="power.power.effective_var"></span>

### `power.power.effective_var`

**Cause and message:**

<p>effective_var must be &gt; 0, got {effective_var}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `effective_var`

**Registering modules:** `increment.power.core`

<span id="power.power_bounds_enclose"></span>

### `power.power_bounds_enclose`

**Cause and message:**

<p>the enclosure must satisfy 0 &lt;= power_lower &lt;= power &lt;= power_upper &lt;= 1, got power_lower={power_lower}, power={power}, power_upper={power_upper}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `power`, `power_lower`, `power_upper`

**Registering modules:** `increment.power.core`

<span id="power.power_bounds_paired"></span>

### `power.power_bounds_paired`

**Cause and message:**

<p>power_lower and power_upper are stated together, got power_lower={power_lower!r}, power_upper={power_upper!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `power_lower`, `power_upper`

**Registering modules:** `increment.power.core`

<span id="power.power_bounds_required"></span>

### `power.power_bounds_required`

**Cause and message:**

<p>power_basis=&#x27;lower_bound&#x27; states the enclosure whose lower end power is: power_lower and power_upper are required</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.power_is_lower_bound"></span>

### `power.power_is_lower_bound`

**Cause and message:**

<p>power_basis=&#x27;lower_bound&#x27; means power is the enclosure&#x27;s lower end, got power={power} and power_lower={power_lower}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `power`, `power_lower`

**Registering modules:** `increment.power.core`

<span id="power.power_solvers_relative"></span>

### `power.power_solvers_relative`

**Cause and message:**

<p>power solvers require a relative ArmPlanningProcedure decision</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power._shared`, `increment.power.core`, `increment.power.pairwise`

<span id="power.procedure_armplanningprocedure"></span>

### `power.procedure_armplanningprocedure`

**Cause and message:**

<p>procedure must be ArmPlanningProcedure, got {procedure_type}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `procedure_type`

**Registering modules:** `increment.power.core`

<span id="power.q_a"></span>

### `power.q_a`

**Cause and message:**

<p>q_a must be in (0, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.pairwise`

<span id="power.q_a_q"></span>

### `power.q_a_q`

**Cause and message:**

<p>q_a + q_b must be &lt;= 1, got {q_a_plus_q_b}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `q_a_plus_q_b`

**Registering modules:** `increment.power.pairwise`

<span id="power.q_b"></span>

### `power.q_b`

**Cause and message:**

<p>q_b must be in (0, 1)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.pairwise`

<span id="power.quantile_baseline.cuped_unsupported"></span>

### `power.quantile_baseline.cuped_unsupported`

**Cause and message:**

<p>cuped_rho={cuped_rho} was declared, but a quantile metric has no runtime CUPED route to credit; QuantileBaseline always plans without variance reduction</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cuped_rho`

**Registering modules:** `increment.power.core`

<span id="power.quantile_baseline.metric_not_quantile"></span>

### `power.quantile_baseline.metric_not_quantile`

**Cause and message:**

<p>a QuantileBaseline plans a quantile metric, but {metric!r} is declared as {metric_type!r}; plan a quantile metric with ArmPlanningProcedure.standard(&#x27;quantile&#x27;), and any other metric with workflows.planning_baseline(experiment, reader, metric) or a Baseline constructor</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `metric_type`

**Registering modules:** `increment.power.core`

<span id="power.quantile_baseline.pilot_values_missing"></span>

### `power.quantile_baseline.pilot_values_missing`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L287).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_name`, `n_values`, `route`

**Registering modules:** `increment.power.core`

<span id="power.quantile_size_search_delta_method_unreliable"></span>

### `power.quantile_size_search_delta_method_unreliable`

**Cause and message:**

<p>no sample size gives this quantile metric a lift the readout can report: wherever power reaches {power}, the pilot&#x27;s recording grid keeps the planned combined log-scale standard error at about {log_se:.4g} or more, and the readout refuses a relative lift whose log-scale standard error is {log_se_limit} or more (delta_method_unreliable); record the metric at a finer resolution</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `log_se`, `log_se_limit`, `power`

**Registering modules:** `increment.power.core`

<span id="power.quantile_size_search_unreachable"></span>

### `power.quantile_size_search_unreachable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/core.py#L297).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `limiting_condition`, `maximum_power`, `power`

**Registering modules:** `increment.power.core`

<span id="power.r_a_above"></span>

### `power.r_a_above`

**Cause and message:**

<p>r_a ({r_a}) is above r_b ({r_b}) (theta={theta:.6g}) but alternative=&#x27;less&#x27;; no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `r_a`, `r_b`, `theta`

**Registering modules:** `increment.power.pairwise`

<span id="power.r_a_below"></span>

### `power.r_a_below`

**Cause and message:**

<p>r_a ({r_a}) is below r_b ({r_b}) (theta={theta:.6g}) but alternative=&#x27;greater&#x27;; no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `r_a`, `r_b`, `theta`

**Registering modules:** `increment.power.pairwise`

<span id="power.r_a_r"></span>

### `power.r_a_r`

**Cause and message:**

<p>r_a ({r_a}) and r_b ({r_b}) give the same relative lift (theta=0); no finite sample size can distinguish segment A from segment B in this design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `r_a`, `r_b`

**Registering modules:** `increment.power.pairwise`

<span id="power.relative_lift_lies"></span>

### `power.relative_lift_lies`

**Cause and message:**

<p>relative_lift={relative_lift} lies below the null boundary (null_lift={null_lift}) but alternative=&#x27;greater&#x27;; no sample size gives this design more than alpha power; flip the alternative or the sign of relative_lift</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`, `relative_lift`

**Registering modules:** `increment.power.core`

<span id="power.relative_lift_lies_above_null"></span>

### `power.relative_lift_lies_above_null`

**Cause and message:**

<p>relative_lift={relative_lift} lies above the null boundary (null_lift={null_lift}) but alternative=&#x27;less&#x27;; no sample size gives this design more than alpha power; flip the alternative or the sign of relative_lift</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`, `relative_lift`

**Registering modules:** `increment.power.core`

<span id="power.require_relative_domain"></span>

### `power.require_relative_domain`

**Cause and message:**

<p>{name} must be &gt; -1.0, got {value}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `value`

**Registering modules:** `increment.power._validation`

<span id="power.sample_size_detecting"></span>

### `power.sample_size_detecting`

**Cause and message:**

<p>the sample size detecting relative_lift={relative_lift} against null_lift={null_lift} at this baseline exceeds the float64 range: the log-scale distance is too small for the planning variance</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`, `relative_lift`

**Registering modules:** `increment.power.core`

<span id="power.scalar_numeric_sequence"></span>

### `power.scalar_numeric_sequence`

**Cause and message:**

<p>{name} must be a scalar or numeric sequence</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.power.curve`

<span id="power.se_full_finite"></span>

### `power.se_full_finite`

**Cause and message:**

<p>se_full must be finite and &gt; 0 (got {se_full!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `se_full`

**Registering modules:** `increment.power.sequential`

<span id="power.segment_clustered_baseline"></span>

### `power.segment_clustered_baseline`

**Cause and message:**

<p>segment {segment}&#x27;s clustered baseline requires at least two clusters per arm (got {k_total} total, n_t={n_t}, n_c={n_c}); increase n_total or the segment&#x27;s share</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `k_total`, `n_c`, `n_t`, `segment`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_pairwise_achieved"></span>

### `power.segment_pairwise_achieved`

**Cause and message:**

<p>segment_pairwise_achieved_power does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_pairwise_achieved_n_per_arm_too_small"></span>

### `power.segment_pairwise_achieved_n_per_arm_too_small`

**Cause and message:**

<p>n_per_arm={n_per_arm} is too small to split segments A (q={q_a}) and B (q={q_b}) into 2-arm designs ({exc}); increase n_per_arm or the smaller segment&#x27;s share</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `exc`, `n_per_arm`, `q_a`, `q_b`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_pairwise_minimum"></span>

### `power.segment_pairwise_minimum`

**Cause and message:**

<p>segment_pairwise_minimum_detectable_effect does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_pairwise_required"></span>

### `power.segment_pairwise_required`

**Cause and message:**

<p>segment_pairwise_required_sample_size does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_pairwise_solvers"></span>

### `power.segment_pairwise_solvers`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/power/pairwise.py#L79).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `unsupported`

**Registering modules:** `increment.power.pairwise`

<span id="power.segment_share_n"></span>

### `power.segment_share_n`

**Cause and message:**

<p>segment share q={q} of n_total={n_total} implies only {n_total_seg:.3g} units, too few for a 2-arm design at allocation={allocation} (need &gt;= {min_n_total_seg:.3g} total at this allocation); increase n_total or the segment&#x27;s share</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allocation`, `min_n_total_seg`, `n_total`, `n_total_seg`, `q`

**Registering modules:** `increment.power.pairwise`

<span id="power.sequential_design_more"></span>

### `power.sequential_design_more`

**Cause and message:**

<p>sequential design requires more than 1024x the fixed-horizon sample size to reach power={power}; check the InferenceSpec(kind=&#x27;asymptotic_mean&#x27;) declared on ArmPlanningProcedure.standard() for a mismatch with the target relative_lift</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `power`

**Registering modules:** `increment.power.core`

<span id="power.sequential_mde.compliance_unrepresentable"></span>

### `power.sequential_mde.compliance_unrepresentable`

**Cause and message:**

<p>sequential design at power={target_power}: the raw minimum detectable relative lift ({mde_relative:.6g}) adjusted onto the complier scale by compliance {compliance:.6g} overflows float64; the compliance is too small for a representable complier-scale effect</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `compliance`, `mde_relative`, `target_power`

**Registering modules:** `increment.power.sequential`

<span id="power.sequential_mde.decrease_unrepresentable"></span>

### `power.sequential_mde.decrease_unrepresentable`

**Cause and message:**

<p>sequential design at power={target_power}: the raw minimum detectable relative lift ({mde_relative:.6g}) adjusted onto the complier scale by compliance {compliance:.6g} reaches or exceeds a 100% decrease, which is not a representable relative effect; the detectable decrease is too large to resolve at this standard error</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `compliance`, `mde_relative`, `target_power`

**Registering modules:** `increment.power.sequential`

<span id="power.sequential_mde.design_search_minimum"></span>

### `power.sequential_mde.design_search_minimum`

**Cause and message:**

<p>sequential design cannot search for a minimum detectable effect when target_power={target} is at or below this boundary&#x27;s own null-crossing probability ({estimate:.6g}); the zero-distance effect already meets it, so no positive relative lift is more detectable than the null itself</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimate`, `target`

**Registering modules:** `increment.power.sequential`

<span id="power.sequential_mde.unrepresentable"></span>

### `power.sequential_mde.unrepresentable`

**Cause and message:**

<p>sequential design at power={target_power}, se={se:.6g}: the minimum detectable log-effect ({mde_theta:.6g}) is too large to express as a relative lift in float64; the standard error is too large for a relative-scale minimum detectable effect</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `mde_theta`, `se`, `target_power`

**Registering modules:** `increment.power.sequential`

<span id="power.sequential_sample_size"></span>

### `power.sequential_sample_size`

**Cause and message:**

<p>sequential sample-size power could not meet its numerical tolerance at n_per_arm={n_per_arm}: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_per_arm`, `reason`

**Registering modules:** `increment.power.core`

<span id="power.sequential_sample_size_target_inside_enclosure"></span>

### `power.sequential_sample_size_target_inside_enclosure`

**Cause and message:**

<p>sequential sample-size target lies inside the certified power enclosure at n_per_arm={n_per_arm}: ({lower}, {upper})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lower`, `n_per_arm`, `upper`

**Registering modules:** `increment.power.core`

<span id="power.size_design_relative"></span>

### `power.size_design_relative`

**Cause and message:**

<p>cannot size a design for a relative_lift exactly at the null boundary (relative_lift={relative_lift}, null_lift={null_lift}): the distance to detect is zero, so no finite sample size reaches any power above alpha; pass a lift away from the null (the same refusal segment_pairwise_required_sample_size makes for theta=0)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `null_lift`, `relative_lift`

**Registering modules:** `increment.power.core`

<span id="power.solved_too_small"></span>

### `power.solved_too_small`

**Cause and message:**

<p>the solved-for N ({n_total}) is too small to split segments A (q={q_a}) and B (q={q_b}) into 2-arm designs ({exc}); this can happen when the contrast between r_a and r_b is large relative to the smaller segment&#x27;s share, needing more of the solved-for N per segment than a 2-arm split allows; narrow the contrast, raise the smaller segment&#x27;s share, or accept that an easy-to-detect contrast needs a manually-chosen larger N</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `exc`, `n_total`, `q_a`, `q_b`

**Registering modules:** `increment.power.pairwise`

<span id="power.supports_fixed_horizon"></span>

### `power.supports_fixed_horizon`

**Cause and message:**

<p>{caller} supports fixed-horizon ArmPlanningProcedure only</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `caller`

**Registering modules:** `increment.power.pairwise`

<span id="power.switchback.baseline"></span>

### `power.switchback.baseline`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.bounded_model_required"></span>

### `power.switchback.bounded_model_required`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.delta"></span>

### `power.switchback.delta`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.n"></span>

### `power.switchback.n`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.procedure"></span>

### `power.switchback.procedure`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.result"></span>

### `power.switchback.result`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.roster"></span>

### `power.switchback.roster`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.switchback.target_power"></span>

### `power.switchback.target_power`

**Cause and message:**

<p>{message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `field`, `message`, `route`, `value`

**Registering modules:** `increment.power.switchback`

<span id="power.tau_b"></span>

### `power.tau_b`

**Cause and message:**

<p>tau_b must be &gt;= 0, got {tau_b}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `tau_b`

**Registering modules:** `increment.power.core`

<span id="power.theta_contains_non"></span>

### `power.theta_contains_non`

**Cause and message:**

<p>theta contains non-finite values</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.core`

<span id="power.theta_var_one"></span>

### `power.theta_var_one`

**Cause and message:**

<p>theta and var must be 1-D (one value per segment), got shapes {theta_shape} and {var_shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `theta_shape`, `var_shape`

**Registering modules:** `increment.power.core`

<span id="power.theta_var_same"></span>

### `power.theta_var_same`

**Cause and message:**

<p>theta and var must have the same shape, got {theta_shape} vs {var_shape}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `theta_shape`, `var_shape`

**Registering modules:** `increment.power.core`

<span id="power.unit_cycle.input"></span>

### `power.unit_cycle.input`

**Cause and message:**

<p>{detail} (field={field!r}, value={value!r}, constraint={constraint!r}; route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `detail`, `field`, `route`, `value`

**Registering modules:** `increment.power.unit_cycle`

<span id="power.unit_cycle.law"></span>

### `power.unit_cycle.law`

**Cause and message:**

<p>{detail} (field={field!r}, value={value!r}, constraint={constraint!r}; route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `detail`, `field`, `route`, `value`

**Registering modules:** `increment.power.unit_cycle`

<span id="power.unit_cycle.numerical"></span>

### `power.unit_cycle.numerical`

**Cause and message:**

<p>{detail} (field={field!r}, value={value!r}, constraint={constraint!r}; route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `detail`, `field`, `route`, `value`

**Registering modules:** `increment.power.unit_cycle`

<span id="power.unit_cycle.procedure"></span>

### `power.unit_cycle.procedure`

**Cause and message:**

<p>{detail} (field={field!r}, value={value!r}, constraint={constraint!r}; route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `detail`, `field`, `route`, `value`

**Registering modules:** `increment.power.unit_cycle`

<span id="power.unit_cycle.result"></span>

### `power.unit_cycle.result`

**Cause and message:**

<p>{detail} (field={field!r}, value={value!r}, constraint={constraint!r}; route={route!r})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constraint`, `detail`, `field`, `route`, `value`

**Registering modules:** `increment.power.unit_cycle`

<span id="power.units_per_week_finite_positive"></span>

### `power.units_per_week_finite_positive`

**Cause and message:**

<p>units_per_week must be a finite positive number</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.power.curve`

<span id="family-query"></span>

## `query.*`

<span id="query.artifact_reader.artifact_moment.manifest_no_metric"></span>

### `query.artifact_reader.artifact_moment.manifest_no_metric`

**Cause and message:**

<p>artifact manifest has no metric bindings</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_reader`

<span id="query.artifact_reader.artifact_moment.verification_lazy_digest"></span>

### `query.artifact_reader.artifact_moment.verification_lazy_digest`

**Cause and message:**

<p>only verification=&#x27;lazy_digest&#x27; is supported</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.artifact_reader`

<span id="query.artifact_reader.snapshot_execution_returned"></span>

### `query.artifact_reader.snapshot_execution_returned`

**Cause and message:**

<p>snapshot execution returned unsupported relation {type_name}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `type_name`

**Registering modules:** `increment.query.artifact_reader`

<span id="query.backend.compiler_unavailable"></span>

### `query.backend.compiler_unavailable`

**Cause and message:**

<p>the installed Ibis cannot write SQL for {backend!r}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `backend`, `route`

**Registering modules:** `increment._compile_gaps`

<span id="query.backend.operation_unsupported"></span>

### `query.backend.operation_unsupported`

**Cause and message:**

<p>the {backend} SQL dialect cannot express this analysis: {detail}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `backend`, `detail`, `route`

**Registering modules:** `increment._compile_gaps`

<span id="query.builders.asof_group_summary"></span>

### `query.builders.asof_group_summary`

**Cause and message:**

<p>asof_group_summary: completed_windows_only=True is contradictory for retention metric &#x27;{name}&#x27;; its band (threshold_days={threshold_days}) is open on the right, so no unit&#x27;s window ever completes and the gate would admit nothing. Drop completed_windows_only to get the cumulative monitoring series, or declare threshold_days: [a, b] to bound the band.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `threshold_days`

**Registering modules:** `increment.query.builders`

<span id="query.builders.asof_group_summary_completed_requires_uptake_window"></span>

### `query.builders.asof_group_summary_completed_requires_uptake_window`

**Cause and message:**

<p>asof_group_summary: completed_windows_only=True under an encouragement design requires bounded outcome and uptake windows</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.asof_group_summary_den_panel_retention"></span>

### `query.builders.asof_group_summary_den_panel_retention`

**Cause and message:**

<p>asof_group_summary: den_panel is meaningless for a RetentionMetric (retention has no denominator events; its denominator is the gated population, which the band-open/maturity gate already produces)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.asof_group_summary_metric_type_not_implemented"></span>

### `query.builders.asof_group_summary_metric_type_not_implemented`

**Cause and message:**

<p>asof_group_summary for {metric_type} not implemented</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.query.builders`

<span id="query.builders.check_cluster.experiment_arm_no"></span>

### `query.builders.check_cluster.experiment_arm_no`

**Cause and message:**

<p>experiment &#x27;{experiment_name}&#x27;: arm &#x27;{group}&#x27; has no clusters in the exposure counts; cannot check cross-arm cluster-size balance.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment_name`, `group`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cluster_column_missing"></span>

### `query.builders.cluster_column_missing`

**Cause and message:**

<p>experiment &#x27;{experiment}&#x27; declares cluster &#x27;{cluster}&#x27;, but the {table} table has no such column; {remedy} Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `columns`, `experiment`, `remedy`, `table`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cluster_size_imbalance"></span>

### `query.builders.cluster_size_imbalance`

**Cause and message:**

<p>experiment &#x27;{experiment}&#x27;: mean cluster size differs by {gap:.0%} across the contrasted arms (cluster &#x27;{cluster}&#x27;: &#x27;{control_group}&#x27; {control_mean_size:.1f} units/cluster vs &#x27;{target_group}&#x27; {target_mean_size:.1f}), above the {threshold:.0%} whole-site refusal threshold. Cluster randomization balances clusters, not units, so under this imbalance the whole-site impact is not identified as the unit-weighted average treatment effect; the differenced ratio-of-sums confounds the treatment effect with the cluster-size gap. Read the per-unit lift from run() instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `control_group`, `control_mean_size`, `experiment`, `gap`, `target_group`, `target_mean_size`, `threshold`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cluster_total_grain"></span>

### `query.builders.cluster_total_grain`

**Cause and message:**

<p>{caller}: {grain} rows cannot combine with the declared cluster &#x27;{cluster}&#x27;; cluster-robust inference is total-grain only. Read the total-grain result instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `caller`, `cluster`, `grain`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cluster_undeclared"></span>

### `query.builders.cluster_undeclared`

**Cause and message:**

<p>experiment &#x27;{experiment}&#x27; declares no cluster; {operation} has no grain to count over. Count units off first_exposures directly instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `experiment`, `operation`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cohort_group_summary"></span>

### `query.builders.cohort_group_summary`

**Cause and message:**

<p>cohort_group_summary for {metric_type} not implemented; the cohort axis exists for retention, whose outcome is not a property of a single activity day; use daily_group_summary for metric types that have an independent per-day reading</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.query.builders`

<span id="query.builders.cohort_group_summary_unbounded_band"></span>

### `query.builders.cohort_group_summary_unbounded_band`

**Cause and message:**

<p>cohort_group_summary: retention metric &#x27;{name}&#x27; declares an unbounded band (threshold_days={threshold_days}), so no cohort&#x27;s outcome ever completes and no cohort could be reported. Declare threshold_days: [a, b] to bound the band.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `threshold_days`

**Registering modules:** `increment.query.builders`

<span id="query.builders.daily_state_not_implemented"></span>

### `query.builders.daily_state_not_implemented`

**Cause and message:**

<p>daily_group_summary state finalization for {metric_type} not implemented</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.query.builders`

<span id="query.builders.group_summary_declared"></span>

### `query.builders.group_summary_declared`

**Cause and message:**

<p>group_summary: declared cluster column &#x27;{cluster}&#x27; is not on the totals table; build totals with unit_totals under the same cluster-declaring experiment. Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `columns`

**Registering modules:** `increment.query.builders`

<span id="query.builders.group_summary_ratio"></span>

### `query.builders.group_summary_ratio`

**Cause and message:**

<p>group_summary: ratio_metrics={ratio_names} names a metric whose denominator must come from a &#x27;y_den&#x27; column, which is not on the totals table. Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `ratio_names`

**Registering modules:** `increment.query.builders`

<span id="query.builders.group_summary_uptake"></span>

### `query.builders.group_summary_uptake`

**Cause and message:**

<p>group_summary: uptake=True declares an encouragement uptake fact in a &#x27;d&#x27; column, which is not on the totals table; build totals with unit_totals(uptake_events=...). Available columns: {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment.query.builders`

<span id="query.builders.invalid_day_boundary"></span>

### `query.builders.invalid_day_boundary`

**Cause and message:**

<p>invalid day_boundary {day_boundary!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `day_boundary`

**Registering modules:** `increment.query.builders`

<span id="query.builders.part_ratiometric"></span>

### `query.builders.part_ratiometric`

**Cause and message:**

<p>part=&#x27;{part}&#x27; is only valid for RatioMetric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `part`

**Registering modules:** `increment.query.builders`

<span id="query.builders.ratiometric_den_stats"></span>

### `query.builders.ratiometric_den_stats`

**Cause and message:**

<p>RatioMetric requires den_stats (denominator sufficient stats)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.site_volume_metric_type"></span>

### `query.builders.site_volume_metric_type`

**Cause and message:**

<p>metric &#x27;{metric}&#x27;: site volume is undefined for a {metric_type} metric; {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `metric_type`, `reason`

**Registering modules:** `increment.query.builders`

<span id="query.builders.trigger_population.cutoff_not_aware"></span>

### `query.builders.trigger_population.cutoff_not_aware`

**Cause and message:**

<p>{field} must be timezone-aware; received {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `value`

**Registering modules:** `increment.query.builders`

<span id="query.builders.union_event_horizon"></span>

### `query.builders.union_event_horizon`

**Cause and message:**

<p>union_event_horizon requires at least one event table</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.unit_totals_by"></span>

### `query.builders.unit_totals_by`

**Cause and message:**

<p>unit_totals: `by` requires `properties_table`</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.unit_totals_properties"></span>

### `query.builders.unit_totals_properties`

**Cause and message:**

<p>unit_totals: `properties_table` given without `by`; nothing to group by</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.builders`

<span id="query.builders.unknown_aggregation"></span>

### `query.builders.unknown_aggregation`

**Cause and message:**

<p>unknown aggregation {aggregation!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `aggregation`

**Registering modules:** `increment.query.builders`, `increment.query.calendar`

<span id="query.builders.unknown_part_ratiometric"></span>

### `query.builders.unknown_part_ratiometric`

**Cause and message:**

<p>Unknown part &#x27;{part}&#x27; for RatioMetric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `part`

**Registering modules:** `increment.query.builders`

<span id="query.builders.window_bound_stats"></span>

### `query.builders.window_bound_stats`

**Cause and message:**

<p>window_bound_stats for {metric_type} not implemented</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.query.builders`

<span id="query.calendar.period_moments_by"></span>

### `query.calendar.period_moments_by`

**Cause and message:**

<p>period_moments &#x27;by&#x27; table must carry exactly one dimension column besides unit_id/period; got {dim_cols}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dim_cols`

**Registering modules:** `increment.query.calendar`

<span id="query.calendar.period_start_unknown_grain"></span>

### `query.calendar.period_start_unknown_grain`

**Cause and message:**

<p>unknown grain {grain!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `grain`

**Registering modules:** `increment.query.calendar`

<span id="query.calendar.population_unit_id"></span>

### `query.calendar.population_unit_id`

**Cause and message:**

<p>population must have a &#x27;unit_id&#x27; column; got {columns}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment.query.calendar`

<span id="query.calendar.range"></span>

### `query.calendar.range`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/calendar.py#L61).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `end_edge`, `span`, `start`

**Registering modules:** `increment.query.calendar`

<span id="query.cuped.general_qr_required"></span>

### `query.cuped.general_qr_required`

**Cause and message:**

<p>Metric {metric!r} requires general warehouse QR ({reason}); {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `reason`, `route`

**Registering modules:** `increment.query._cuped_cells`

<span id="query.cuped.magnitude_range"></span>

### `query.cuped.magnitude_range`

**Cause and message:**

<p>Metric {metric!r} has {component} in arm {arm!r} outside the warehouse CUPED RMS magnitude range [1e-50, 1e50] (bound: {bound}); rescale the values (change units) or fit from explicitly supplied per-unit data with from_frame.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `arm`, `bound`, `component`, `metric`

**Registering modules:** `increment.query._cuped_cells`

<span id="query.dims.dim_source_fact"></span>

### `query.dims.dim_source_fact`

**Cause and message:**

<p>dim source &#x27;{name}&#x27;: the fact source projects a &#x27;valid_from&#x27;/&#x27;valid_to&#x27; column, which the generated range join reserves; alias it in the fact source SQL</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`

**Registering modules:** `increment.query.dims`

<span id="query.dims.dim_source_multiple"></span>

### `query.dims.dim_source_multiple`

**Cause and message:**

<p>dim source &#x27;{name}&#x27;: multiple rows for the same &#x27;{entity}&#x27; value would fan out the fact table; deduplicate the dim source or declare a validity range</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `entity`, `name`

**Registering modules:** `increment.query.dims`

<span id="query.dims.dim_source_overlapping"></span>

### `query.dims.dim_source_overlapping`

**Cause and message:**

<p>dim source &#x27;{name}&#x27;: overlapping validity ranges for the same &#x27;{entity}&#x27; value; fix the changelog/validity source</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `entity`, `name`

**Registering modules:** `increment.query.dims`

<span id="query.dims.dim_source_property"></span>

### `query.dims.dim_source_property`

**Cause and message:**

<p>dim source &#x27;{name}&#x27;: property &#x27;{column}&#x27; collides with an existing column on the fact source; rename the property or drop the fact-side column from the source SQL</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `name`

**Registering modules:** `increment.query.dims`

<span id="query.dims.dim_source_two"></span>

### `query.dims.dim_source_two`

**Cause and message:**

<p>dim source &#x27;{name}&#x27;: two rows share one validity boundary for the same &#x27;{entity}&#x27; value; the tie would resolve by physical row order rather than the declared history</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `entity`, `name`

**Registering modules:** `increment.query.dims`

<span id="query.fact_resolution.breakout_property_could"></span>

### `query.fact_resolution.breakout_property_could`

**Cause and message:**

<p>breakout property &#x27;{property}&#x27; could not be resolved to any fact source with unit &#x27;{unit}&#x27; as an entity; this should have been caught at definitions-load time</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `property`, `unit`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.breakout_property_references"></span>

### `query.fact_resolution.breakout_property_references`

**Cause and message:**

<p>breakout property &#x27;{property}&#x27; references unknown source &#x27;{source}&#x27;; this should have been caught at definitions-load time</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `property`, `source`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.fact_declared_any"></span>

### `query.fact_resolution.fact_declared_any`

**Cause and message:**

<p>Fact &#x27;{fact_name}&#x27; is not declared in any fact source in the definitions. Available facts: {available}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `fact_name`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.fact_source_property"></span>

### `query.fact_resolution.fact_source_property`

**Cause and message:**

<p>fact source &#x27;{name}&#x27;: property &#x27;{prop_name}&#x27; renames onto the canonical/value column &#x27;{prop_name}&#x27;, which would silently overwrite it; rename the property</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `prop_name`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.fact_source_property_deletes_value_column"></span>

### `query.fact_resolution.fact_source_property_deletes_value_column`

**Cause and message:**

<p>fact source &#x27;{name}&#x27;: property &#x27;{prop_name}&#x27; renames away the canonical/value column &#x27;{prop_column}&#x27;, which would silently delete it; point the property at a different column</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `prop_column`, `prop_name`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.fact_source_property_reads_canonical_column"></span>

### `query.fact_resolution.fact_source_property_reads_canonical_column`

**Cause and message:**

<p>fact source &#x27;{name}&#x27;: property &#x27;{prop_name}&#x27; reads the canonical source column &#x27;{prop_column}&#x27;, which the builders rename onto ts/unit_id; point the property at a different column</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `prop_column`, `prop_name`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.fact_resolution.pass_exactly_one"></span>

### `query.fact_resolution.pass_exactly_one`

**Cause and message:**

<p>{constructor}: pass exactly one of control= or design=; control= is shorthand for design=Randomized(control_group=...); got control={control!r}, design={design!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructor`, `control`, `design`

**Registering modules:** `increment.query.fact_resolution`

<span id="query.integrity.cluster_conflict"></span>

### `query.integrity.cluster_conflict`

**Cause and message:**

<p>experiment {experiment!r}: cluster {cluster!r} has multiple non-null labels for enrolled unit(s) {units!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `experiment`, `units`

**Registering modules:** `increment.query.integrity`

<span id="query.integrity.cluster_labels"></span>

### `query.integrity.cluster_labels`

**Cause and message:**

<p>experiment {experiment!r}: cluster field {cluster!r} has invalid {kind}: value/count {value!r}; constraint {constraint}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `constraint`, `experiment`, `kind`, `value`

**Registering modules:** `increment.query.integrity`

<span id="query.integrity.mixed_assignment_units"></span>

### `query.integrity.mixed_assignment_units`

**Cause and message:**

<p>{finding}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `finding`, `mixed_count`, `route`, `unassigned_count`

**Registering modules:** `increment.query.integrity`

<span id="query.integrity.trigger_arm_missing"></span>

### `query.integrity.trigger_arm_missing`

**Cause and message:**

<p>trigger {trigger!r} never fires in arm(s) {silent!r} (observed rates: {rates!r}); {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `rates`, `route`, `silent`, `trigger`

**Registering modules:** `increment._trigger_integrity`

<span id="query.integrity.unassigned_assignment_units"></span>

### `query.integrity.unassigned_assignment_units`

**Cause and message:**

<p>{finding}; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `finding`, `mixed_count`, `route`, `unassigned_count`

**Registering modules:** `increment.query.integrity`

<span id="query.relation.name_occupied"></span>

### `query.relation.name_occupied`

**Cause and message:**

<p>warehouse relation {name!r} already exists; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `backend`, `catalog`, `name`, `route`, `schema`

**Registering modules:** `increment.query.session`

<span id="query.session.artifact_publication.handle_closed"></span>

### `query.session.artifact_publication.handle_closed`

**Cause and message:**

<p>publication handle is closed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.materialization_namespace"></span>

### `query.session.materialization_namespace`

**Cause and message:**

<p>TEMP namespace could not be resolved for {backend}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `backend`

**Registering modules:** `increment.query.session`

<span id="query.session.qr_scratch_cleanup_failed"></span>

### `query.session.qr_scratch_cleanup_failed`

**Cause and message:**

<p>warehouse QR scratch cleanup failed; retained relations are listed for manual removal: {relations}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `relations`, `route`

**Registering modules:** `increment.query.session`

<span id="query.session.snapshot.cutoff_not_aware"></span>

### `query.session.snapshot.cutoff_not_aware`

**Cause and message:**

<p>{field} must be timezone-aware; received {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `value`

**Registering modules:** `increment.query.session`

<span id="query.session.snapshot.feed_name_invalid"></span>

### `query.session.snapshot.feed_name_invalid`

**Cause and message:**

<p>feed names must be non-empty strings without NUL; received {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment.query.session`

<span id="query.session.snapshot.materialization_failed"></span>

### `query.session.snapshot.materialization_failed`

**Cause and message:**

<p>this analysis needs warehouse TEMP tables, which this connection cannot create; when only the source snapshot needs them, coherence=&#x27;live&#x27; runs without one, otherwise analyze from a dataframe with from_frame</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.snapshot.source_missing"></span>

### `query.session.snapshot.source_missing`

**Cause and message:**

<p>source was not included in the pinned execution; a live fallback is forbidden</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.snapshot.watermark_not_aware"></span>

### `query.session.snapshot.watermark_not_aware`

**Cause and message:**

<p>complete-through watermark for feed {feed!r} ({field}) must be timezone-aware; received {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `feed`, `field`, `value`

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.manifest_bound_publication"></span>

### `query.session.warehouse_artifact.manifest_bound_publication`

**Cause and message:**

<p>manifest is not bound to this publication</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.manifest_context_differs"></span>

### `query.session.warehouse_artifact.manifest_context_differs`

**Cause and message:**

<p>manifest context differs from publication context</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.manifest_digest_does"></span>

### `query.session.warehouse_artifact.manifest_digest_does`

**Cause and message:**

<p>manifest digest does not match manifest body</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.manifest_references_unwritten"></span>

### `query.session.warehouse_artifact.manifest_references_unwritten`

**Cause and message:**

<p>manifest references an unwritten relation</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.manifest_relation_locators"></span>

### `query.session.warehouse_artifact.manifest_relation_locators`

**Cause and message:**

<p>manifest relation locators must be unique</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.publication_cleanup_incomplete"></span>

### `query.session.warehouse_artifact.publication_cleanup_incomplete`

**Cause and message:**

<p>publication ended without a manifest and some relations could not be dropped; the manifest insert never began, so drop the listed relations by name</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `artifact_id`, `generation_id`, `relations`, `route`

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.publication_handle_sealed"></span>

### `query.session.warehouse_artifact.publication_handle_sealed`

**Cause and message:**

<p>publication handle is sealed</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.publication_state_unknown"></span>

### `query.session.warehouse_artifact.publication_state_unknown`

**Cause and message:**

<p>publication ended without a manifest and its cleanup did not complete; the retained relations and the recovery route are in the refusal context</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `artifact_id`, `generation_id`, `relations`, `route`, `tombstoned`

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.unknown_digest_format"></span>

### `query.session.warehouse_artifact.unknown_digest_format`

**Cause and message:**

<p>unknown artifact relation digest format {digest_format!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `digest_format`

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.unknown_relation_role"></span>

### `query.session.warehouse_artifact.unknown_relation_role`

**Cause and message:**

<p>unknown artifact relation role {role!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `role`

**Registering modules:** `increment.query.session`

<span id="query.session.warehouse_artifact.verification_unavailable"></span>

### `query.session.warehouse_artifact.verification_unavailable`

**Cause and message:**

<p>artifact relation {relation!r} uses digest format {digest_format}, which cannot be verified warehouse-side; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `digest_format`, `relation`, `route`

**Registering modules:** `increment.query.session`

<span id="family-readout"></span>

## `readout.*`

<span id="readout.adjustment.absolute_unadjusted"></span>

### `readout.adjustment.absolute_unadjusted`

**Cause and message:**

<p>Method(name=&#x27;unadjusted&#x27;) cannot honor value_scale=&#x27;absolute&#x27;: the unadjusted moments path already reports the absolute pair alongside relative lift, and its rows are confounded</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.plan`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.alwaysvalid_under_encouragement"></span>

### `readout.alwaysvalid_under_encouragement`

**Cause and message:**

<p>{method}: AlwaysValid under an encouragement design requires completed_windows_only=True so repeated looks use finalized observations</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `method`

**Registering modules:** `increment.readouts._common`

<span id="readout.arms.no_treatment"></span>

### `readout.arms.no_treatment`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L137).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_group`, `observed_arms`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.arms.none_enrolled"></span>

### `readout.arms.none_enrolled`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L147).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control_group`, `counts`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.asof.cluster"></span>

### `readout.asof.cluster`

**Cause and message:**

<p>asof_lift() is not supported with a declared cluster (&#x27;{cluster}&#x27;): the day-axis views have no cluster-robust variance and would silently report unit-grain uncertainty; cluster-robust inference is total-grain only (run()/srm())</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.asof.segment_unsupported"></span>

### `readout.asof.segment_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L157).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `by`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.assignment.estimands"></span>

### `readout.assignment.estimands`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L62).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `estimands`, `mechanism`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.cell.key_invalid"></span>

### `readout.cell.key_invalid`

**Cause and message:**

<p>Invalid {kind} readout cell fields: {fields}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fields`, `kind`

**Registering modules:** `increment.declarations`, `increment.estimation.readout_types`

<span id="readout.cell.missing_arm"></span>

### `readout.cell.missing_arm`

**Cause and message:**

<p>Metric {metric} has no observations for requested arm {group_id} (observed arms: {observed_arms}); check the observed group label and control spelling, then correct the source or requested contrast and rerun</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `metric`, `observed_arms`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.cell.missing_metric_observations"></span>

### `readout.cell.missing_metric_observations`

**Cause and message:**

<p>Metric {metric} has no usable observations for arm {unobserved_arm} (arms with observations: {observed_arms}); check the metric&#x27;s missing-value policy and the values available in that arm, then correct the data or policy and rerun</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `observed_arms`, `unobserved_arm`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.cell.unsupported_request"></span>

### `readout.cell.unsupported_request`

**Cause and message:**

<p>Requested cell is unsupported: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.collection.concat_cell_collision"></span>

### `readout.collection.concat_cell_collision`

**Cause and message:**

<p>Conflicting cell {cell_key} across snapshots {snapshot_ids}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell_key`, `snapshot_ids`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.collection.concat_metadata_mismatch"></span>

### `readout.collection.concat_metadata_mismatch`

**Cause and message:**

<p>Metadata differs for snapshots {snapshot_ids}: {fields}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fields`, `snapshot_ids`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.collection.concat_type_mismatch"></span>

### `readout.collection.concat_type_mismatch`

**Cause and message:**

<p>Cannot concatenate {left} and {right}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `left`, `right`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.collection.duplicate_cell_key"></span>

### `readout.collection.duplicate_cell_key`

**Cause and message:**

<p>Duplicate cell {cell_key} in source {source_snapshot_id}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell_key`, `source_snapshot_id`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.collection.mutation_unsupported"></span>

### `readout.collection.mutation_unsupported`

**Cause and message:**

<p>Collection mutation {operation} is unsupported for {model}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `model`, `operation`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.completed_encouragement_inference"></span>

### `readout.completed_encouragement_inference`

**Cause and message:**

<p>{method}: completed encouragement inference requires bounded outcome and uptake windows; unbounded metrics={unbound!r}, uptake_window_days={uptake_window_days!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `method`, `unbound`, `uptake_window_days`

**Registering modules:** `increment.readouts._common`

<span id="readout.contrast.request"></span>

### `readout.contrast.request`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_evidence_dispatch.py#L31).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment._evidence_dispatch`

<span id="readout.correction.invalid"></span>

### `readout.correction.invalid`

**Cause and message:**

<p>unknown correction={correction!r}; supported: &#x27;bh&#x27;, &#x27;bonferroni&#x27;, &#x27;none&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `correction`

**Registering modules:** `increment.plan`

<span id="readout.design.required"></span>

### `readout.design.required`

**Cause and message:**

<p>{view}() requires a source with a declared design; this source was constructed without one (design=None). Pass design= at construction (or control=/control_group= to derive Randomized).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `view`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.encouragement.asof_completion"></span>

### `readout.encouragement.asof_completion`

**Cause and message:**

<p>{view}: AlwaysValid under an encouragement design requires completed_windows_only=True so repeated looks use finalized observations</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `view`

**Registering modules:** `increment.estimation.encouragement`

<span id="readout.encouragement.asof_unbounded"></span>

### `readout.encouragement.asof_unbounded`

**Cause and message:**

<p>as-of completed encouragement inference requires bounded outcome and uptake windows; unbounded metrics={metrics!r}, uptake_window_days={uptake_window_days!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metrics`, `uptake_window_days`

**Registering modules:** `increment.estimation.encouragement`

<span id="readout.encouragement.cluster_ratio"></span>

### `readout.encouragement.cluster_ratio`

**Cause and message:**

<p>ratio metric(s) {names} are not supported with a declared cluster (&#x27;{cluster}&#x27;) under an encouragement design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `names`

**Registering modules:** `increment.estimation.encouragement`

<span id="readout.encouragement.correction"></span>

### `readout.encouragement.correction`

**Cause and message:**

<p>multiplicity correction is not supported under an encouragement design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.plan`

<span id="readout.encouragement.margin"></span>

### `readout.encouragement.margin`

**Cause and message:**

<p>metric(s) {names} declare a non-inferiority margin (Metric.margin/margin_abs, or a plan-bound ExperimentMetric.margin), which applies to the ITT row under an encouragement design; this request excludes &#x27;itt&#x27;, so nothing would apply it; add &#x27;itt&#x27; to estimands</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `names`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.encouragement.retention"></span>

### `readout.encouragement.retention`

**Cause and message:**

<p>retention metric(s) {names!r} are not supported under an encouragement design; its maturity gate changes the first-stage population by outcome metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `names`

**Registering modules:** `increment.estimation.encouragement`, `increment.semantics.design`

<span id="readout.encouragement.value_scale"></span>

### `readout.encouragement.value_scale`

**Cause and message:**

<p>value_scale= is an observational-only reporting selector; an encouragement design&#x27;s LATE rows already report the additive estimand in the outcome&#x27;s own units (value_scale=&#x27;absolute&#x27;); select it with estimands= instead.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.estimation.encouragement`, `increment.readouts._common`

<span id="readout.estimands.unknown"></span>

### `readout.estimands.unknown`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L73).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `supported`, `unknown`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.failure.hypothesis_key_collision"></span>

### `readout.failure.hypothesis_key_collision`

**Cause and message:**

<p>Failure hypothesis {hypothesis} maps to multiple cells {cell_keys}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell_keys`, `hypothesis`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.inference.disjoint_slices"></span>

### `readout.inference.disjoint_slices`

**Cause and message:**

<p>run_daily_lift: per-day/per-cohort slices are disjoint, so a sequential guarantee ({inference}) cannot apply to them; an always-valid sequence does apply to the cumulative as-of view, so pass view=&#x27;asof&#x27; (View.asof() with run()), or drop inference for per-slice fixed-horizon intervals (each valid for one planned analysis, not repeated looks)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`, `view`

**Registering modules:** `increment.breakout.estimates`

<span id="readout.inference.observational_run"></span>

### `readout.inference.observational_run`

**Cause and message:**

<p>sequential inference is not supported under an observational design; {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `route`

**Registering modules:** `increment.plan`

<span id="readout.inference.sequential_view"></span>

### `readout.inference.sequential_view`

**Cause and message:**

<p>sequential inference is not supported on the segmented {view} view</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `view`

**Registering modules:** `increment.estimation.encouragement`

<span id="readout.legacy.sampling_unreconstructible"></span>

### `readout.legacy.sampling_unreconstructible`

**Cause and message:**

<p>Legacy sampling cannot be reconstructed for {cell_key}: {missing_fields}; {remedy}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell_key`, `missing_fields`, `remedy`

**Registering modules:** `increment.estimation._readout_refusals`, `increment.estimation.readout_types`

<span id="readout.legacy.scope_unreconstructible"></span>

### `readout.legacy.scope_unreconstructible`

**Cause and message:**

<p>Legacy scope cannot be reconstructed for {source}: {missing_fields}; {remedy}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `missing_fields`, `remedy`, `source`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.margin.breakout"></span>

### `readout.margin.breakout`

**Cause and message:**

<p>metric(s) {names} declare a margin on an encouragement or multi-covariate linear-CUPED breakout; a shifted segment construction is not implemented for these variants. Use the whole-window run() guardrail read, or a supported randomized scalar segment method.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `names`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.metric.percentile_winsorization"></span>

### `readout.metric.percentile_winsorization`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L41).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `names`, `view`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.metric.quantile_alternative"></span>

### `readout.metric.quantile_alternative`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L109).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.metric.quantile_breakout"></span>

### `readout.metric.quantile_breakout`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L100).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.metric.quantile_grain"></span>

### `readout.metric.quantile_grain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sources.py#L377).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `grain`, `metric`

**Registering modules:** `increment.sources`

<span id="readout.metric_declare_non"></span>

### `readout.metric_declare_non`

**Cause and message:**

<p>metric(s) {combined} declare a non-inferiority margin (Metric.margin/margin_abs, or a plan-bound ExperimentMetric.margin), but a per-metric shifted null is not built for the as-of encouragement view; run() applies the margin to the whole-window ITT row, or declare a one-sided plan alternative for a test against null_lift=0.0</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `combined`

**Registering modules:** `increment.readouts._common`

<span id="readout.observational.prior"></span>

### `readout.observational.prior`

**Cause and message:**

<p>mixture priors are only supported on the relative (log-RR) lift scale served by infer_lift/estimate_lift</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.plan`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.observational.quantile"></span>

### `readout.observational.quantile`

**Cause and message:**

<p>quantile metric {metric!r} has no observational estimator: the distribution-free order-statistic interval assumes independently randomized arms, so a confounded contrast would be reported as a causal quantile lift. Run a quantile metric under a randomized design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.randomized.value_scale"></span>

### `readout.randomized.value_scale`

**Cause and message:**

<p>value_scale= is an observational-only reporting selector; the randomized path already reports the absolute axis on every row, and a row whose relative scale is undefined (a non-positive arm mean) is decided on that additive axis automatically.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.request.readoutrequest_requires_one_resolved"></span>

### `readout.request.readoutrequest_requires_one_resolved`

**Cause and message:**

<p>ReadoutRequest requires one resolved metric configuration per selected metric</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.roster.undeclared_observed_arms"></span>

### `readout.roster.undeclared_observed_arms`

**Cause and message:**

<p>allocation declares arms {declared_arms!r} but source assignment evidence contains undeclared arms {undeclared_arms!r} for population {analysis_population!r}; declare every assigned arm in allocation, or omit allocation to use source-observed arms</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `analysis_population`, `declared_arms`, `undeclared_arms`

**Registering modules:** `increment.readouts._common`

<span id="readout.run.segment_unsupported"></span>

### `readout.run.segment_unsupported`

**Cause and message:**

<p>metric {metric!r}: run(by=...) is not supported; the whole-window lift-cell estimator indexes rows by (metric, arm) only, so multiple segments would silently overwrite one another and the result type carries no segment identity; use breakout() or a View.breakout(...) view of run() for a per-segment split instead</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.scalar_method_evidence"></span>

### `readout.scalar_method_evidence`

**Cause and message:**

<p>source route {route!r} cannot produce scalar evidence for method {method!r}; use a unit-bearing source or a source that preserves method-scoped moments</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`, `route`

**Registering modules:** `increment._analysis_config`

<span id="readout.scope.cell_incomplete"></span>

### `readout.scope.cell_incomplete`

**Cause and message:**

<p>cannot project incomplete readout cell: {cell!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell`

**Registering modules:** `increment.readouts._common`

<span id="readout.scope.cell_unresolved"></span>

### `readout.scope.cell_unresolved`

**Cause and message:**

<p>expected readout cell has neither an estimate nor a failure: {cell!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cell`

**Registering modules:** `increment.readouts._common`

<span id="readout.scope.invalid_axis"></span>

### `readout.scope.invalid_axis`

**Cause and message:**

<p>readout scope received unsupported {axis}={value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `axis`, `value`

**Registering modules:** `increment.readouts._common`

<span id="readout.scope.metric_undeclared"></span>

### `readout.scope.metric_undeclared`

**Cause and message:**

<p>readout row names metric {metric!r}, which the metrics in scope {declared!r} do not declare</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared`, `metric`

**Registering modules:** `increment.readouts._common`

<span id="readout.scope.request_not_canonical"></span>

### `readout.scope.request_not_canonical`

**Cause and message:**

<p>Request cannot be canonicalized: {fields}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `fields`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.scope.source_digest_unavailable"></span>

### `readout.scope.source_digest_unavailable`

**Cause and message:**

<p>No source digest for {source} via {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `route`, `source`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.sequential_inference_supported"></span>

### `readout.sequential_inference_supported`

**Cause and message:**

<p>sequential inference is not supported on the whole-window run() entry point under an encouragement design; for sequential monitoring of an encouragement rollout, use asof_lift() instead, which carries time-uniform intervals on its ITT and compliance rows (binary-uptake LATE has no sequential interval; request estimands=(&#x27;itt&#x27;,) or (&#x27;compliance&#x27;,)); declare inference=InferenceSpec(kind=&#x27;always_valid&#x27;, registration=...) on the source&#x27;s AnalysisPlan (asof_lift() reads it from there, not a call kwarg)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.readouts._common`

<span id="readout.serialization.duplicate_cell_key"></span>

### `readout.serialization.duplicate_cell_key`

**Cause and message:**

<p>Duplicate serialized cell {cell_key} in {source_snapshot_id}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `cell_key`, `source_snapshot_id`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.duplicate_object_key"></span>

### `readout.serialization.duplicate_object_key`

**Cause and message:**

<p>Duplicate JSON object key {key}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `key`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.identity_mismatch"></span>

### `readout.serialization.identity_mismatch`

**Cause and message:**

<p>Serialized identity mismatch for {source_snapshot_id}: {reason} ({cell_key})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `cell_key`, `reason`, `source_snapshot_id`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.model_mismatch"></span>

### `readout.serialization.model_mismatch`

**Cause and message:**

<p>Collection {collection} does not contain row model {row_model}{route_suffix}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `collection`, `route_suffix`, `row_model`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.noncanonical_order"></span>

### `readout.serialization.noncanonical_order`

**Cause and message:**

<p>Serialized {section} is not canonically ordered</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `section`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.sequential_snapshot_required"></span>

### `readout.serialization.sequential_snapshot_required`

**Cause and message:**

<p>Serialized checkpoints require sequential snapshot {checkpoint_ids}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `checkpoint_ids`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.serialization.unsupported_version"></span>

### `readout.serialization.unsupported_version`

**Cause and message:**

<p>Unsupported envelope {kind} version {received}; supported {supported}. {remedy}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `kind`, `received`, `remedy`, `supported`

**Registering modules:** `increment.estimation.readout_types`

<span id="readout.srm_source_declared"></span>

### `readout.srm_source_declared`

**Cause and message:**

<p>srm() requires a source with a declared design; this source was constructed without one (design=None). Pass design= at construction (or control=/control_group= to derive Randomized).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.readouts._common`

<span id="readout.value_scale.invalid"></span>

### `readout.value_scale.invalid`

**Cause and message:**

<p>value_scale[{metric!r}]={value_scale!r} must be &#x27;relative&#x27; or &#x27;absolute&#x27;</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `value_scale`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.plan`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.value_scale.null"></span>

### `readout.value_scale.null`

**Cause and message:**

<p>value_scale=&#x27;absolute&#x27; cannot combine with a non-zero or absolute null for metric {metric!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.plan`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.value_scale.unknown_metric"></span>

### `readout.value_scale.unknown_metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/_readout_refusals.py#L91).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `selected`, `unknown`

**Registering modules:** `increment._readout_request`, `increment.breakout.estimates`, `increment.estimation._readout_refusals`, `increment.estimation.adjust`, `increment.estimation.encouragement`, `increment.estimation.engine`, `increment.power.core`, `increment.power.pairwise`

<span id="readout.value_scale_names"></span>

### `readout.value_scale_names`

**Cause and message:**

<p>value_scale= names metrics this source does not report on this call: {unknown_value_scale!r} (selected: {metric_names!r}); refusing rather than silently dropping the request</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_names`, `unknown_value_scale`

**Registering modules:** `increment.readouts._common`

<span id="readout.value_scale_randomized_absolute"></span>

### `readout.value_scale_randomized_absolute`

**Cause and message:**

<p>value_scale= is an observational-only reporting selector; the randomized path already reports the absolute axis (abs_diff/abs_se/abs_lb/abs_ub) on every row, and a row whose relative scale is undefined (a non-positive arm mean) is decided on that additive axis automatically.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** —

**Registering modules:** `increment.readouts._common`

<span id="readout.view.observational"></span>

### `readout.view.observational`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/adjust.py#L163).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `view`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="readout.weight_diagnostics.not_applicable"></span>

### `readout.weight_diagnostics.not_applicable`

**Cause and message:**

<p>weight diagnostics are not applicable to method {method!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `method`

**Registering modules:** `increment.estimation.adjust`, `increment.plan`

<span id="family-report"></span>

## `report.*`

<span id="report.metric.unsupported"></span>

### `report.metric.unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/reporting.py#L76).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `errors`

**Registering modules:** `increment.reporting`

<span id="family-reporting"></span>

## `reporting.*`

<span id="reporting.alpha_finite"></span>

### `reporting.alpha_finite`

**Cause and message:**

<p>alpha must be finite and in (0, 1); got {alpha!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `alpha`

**Registering modules:** `increment.reporting`

<span id="reporting.metric_trend.unknown_backend"></span>

### `reporting.metric_trend.unknown_backend`

**Cause and message:**

<p>unknown backend {backend!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `backend`

**Registering modules:** `increment.reporting`

<span id="reporting.week_start_monday"></span>

### `reporting.week_start_monday`

**Cause and message:**

<p>week_start must be &#x27;monday&#x27; or &#x27;sunday&#x27;; got {week_start!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `week_start`

**Registering modules:** `increment.reporting`

<span id="family-results"></span>

## `results.*`

<span id="results.frame.columns_string"></span>

### `results.frame.columns_string`

**Cause and message:**

<p>to_frame(columns={columns!r}) takes a list of column names; pass {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `route`

**Registering modules:** `increment.breakout.projection`

<span id="results.frame.empty_columns"></span>

### `results.frame.empty_columns`

**Cause and message:**

<p>to_frame(columns=...) requires at least one projection column</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.breakout.projection`

<span id="results.frame.summary_unavailable"></span>

### `results.frame.summary_unavailable`

**Cause and message:**

<p>to_frame(columns=&#x27;summary&#x27;) is not available for {model}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `model`

**Registering modules:** `increment.breakout.projection`

<span id="results.frame.unknown_columns"></span>

### `results.frame.unknown_columns`

**Cause and message:**

<p>to_frame(): unknown column(s) {unknown!r}; available columns are {available!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `unknown`

**Registering modules:** `increment.breakout.projection`

<span id="results.readout.not_retained"></span>

### `results.readout.not_retained`

**Cause and message:**

<p>run {run_id} keeps only its saved result cells; read its cells or envelope, or run the experiment again for its result collections</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `run_id`

**Registering modules:** `increment._run`

<span id="results.readout.scope_required"></span>

### `results.readout.scope_required`

**Cause and message:**

<p>run {run_id} holds several result views; pass one of them to readout(view)</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `run_id`, `scope_hashes`

**Registering modules:** `increment._run`

<span id="results.readout.scope_unknown"></span>

### `results.readout.scope_unknown`

**Cause and message:**

<p>run {run_id} did not execute the requested view; pass a view the run selected</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `run_id`, `scope_hash`

**Registering modules:** `increment._run`

<span id="family-sensitivity"></span>

## `sensitivity.*`

<span id="sensitivity.confounding.estimand_unsupported"></span>

### `sensitivity.confounding.estimand_unsupported`

**Cause and message:**

<p>Confounding sensitivity does not cover estimand={estimand!r} from method={method!r}. Use observational IPTW or AIPW for a marginal risk ratio; randomized contrasts do not require an omitted-confounder bound.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `estimand`, `method`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.metric_type_unavailable"></span>

### `sensitivity.confounding.metric_type_unavailable`

**Cause and message:**

<p>Confounding sensitivity requires the row&#x27;s declared metric type, and metric={metric!r} carries none. Re-run the analysis with a version that records metric type.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.outcome_unsupported"></span>

### `sensitivity.confounding.outcome_unsupported`

**Cause and message:**

<p>Confounding sensitivity requires a binary outcome (a conversion or retention metric); metric_type={metric_type!r} is not supported.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `metric_type`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.ratio_not_representable"></span>

### `sensitivity.confounding.ratio_not_representable`

**Cause and message:**

<p>The sensitivity ratio or equal-strength value is not representable for observed_risk_ratio={observed_risk_ratio!r} and threshold_risk_ratio={threshold_risk_ratio!r}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `observed_risk_ratio`, `threshold_risk_ratio`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.risk_ratio_unavailable"></span>

### `sensitivity.confounding.risk_ratio_unavailable`

**Cause and message:**

<p>Confounding sensitivity requires an available positive finite risk ratio; relative_unavailable_reason={relative_unavailable_reason!r}. Recompute an observational relative estimate with an available point.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `relative_unavailable_reason`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.scale_unsupported"></span>

### `sensitivity.confounding.scale_unsupported`

**Cause and message:**

<p>Confounding sensitivity requires value_scale=&#x27;relative&#x27;; value_scale={value_scale!r} is not supported. Recompute the estimate on the relative scale.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `value_scale`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.sequential_unsupported"></span>

### `sensitivity.confounding.sequential_unsupported`

**Cause and message:**

<p>Confounding sensitivity requires fixed-horizon inference; inference={inference!r} is not supported. Use a fixed-horizon observational estimate.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `UnsupportedRequestError`

**Context keys:** `inference`

**Registering modules:** `increment.estimation.confounding`

<span id="sensitivity.confounding.threshold_invalid"></span>

### `sensitivity.confounding.threshold_invalid`

**Cause and message:**

<p>threshold_risk_ratio={threshold_risk_ratio!r} must be finite and strictly positive.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `threshold_risk_ratio`

**Registering modules:** `increment.estimation.confounding`

<span id="family-sequential"></span>

## `sequential.*`

<span id="sequential.asymptotic_joint.invalid"></span>

### `sequential.asymptotic_joint.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/asymptotic_joint.py#L111).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.asymptotic_joint`

<span id="sequential.asymptotic_mean.invalid"></span>

### `sequential.asymptotic_mean.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/asymptotic_mean.py#L18).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.asymptotic_mean`

<span id="sequential.capture.required"></span>

### `sequential.capture.required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L60).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`, `route`

**Registering modules:** `increment.sequential_state`

<span id="sequential.compliance.segmented_family_unsupported"></span>

### `sequential.compliance.segmented_family_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/sequential.py#L32).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `alternatives`, `family`, `route`, `segments`

**Registering modules:** `increment.semantics.sequential`

<span id="sequential.continuation.legacy"></span>

### `sequential.continuation.legacy`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/sequential.py#L19).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.sequential`, `increment.sequential_state`

<span id="sequential.continuation.rewrite"></span>

### `sequential.continuation.rewrite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.freeze.dropped"></span>

### `sequential.freeze.dropped`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.freeze.invalid"></span>

### `sequential.freeze.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.memo.invalid"></span>

### `sequential.memo.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/estimation/sequential_result.py#L205).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.estimation.sequential_result`

<span id="sequential.registration.invalid"></span>

### `sequential.registration.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/sequential.py#L16).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.semantics.sequential`

<span id="sequential.registration.mixed_requires_asymptotic_mean"></span>

### `sequential.registration.mixed_requires_asymptotic_mean`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/sequential.py#L24).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metrics`

**Registering modules:** `increment.estimation.sequential`, `increment.semantics.sequential`

<span id="sequential.route.unsupported"></span>

### `sequential.route.unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.source.invalid"></span>

### `sequential.source.invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.transform.unpredictable"></span>

### `sequential.transform.unpredictable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/sequential_state.py#L46).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`

**Registering modules:** `increment.sequential_state`

<span id="sequential.wire.rational_invalid"></span>

### `sequential.wire.rational_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/semantics/rational.py#L72).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `field`, `limit_digits`, `reason`, `route_forward`

**Registering modules:** `increment.decision_wire`, `increment.semantics.rational`

<span id="family-simulate"></span>

## `simulate.*`

<span id="simulate.bandit_loggers.count_positive"></span>

### `simulate.bandit_loggers.count_positive`

**Cause and message:**

<p>{name} must be an integer &gt;= 1, got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `name`, `value`

**Registering modules:** `increment.simulate.bandit_loggers`

<span id="simulate.bandit_loggers.epsilon"></span>

### `simulate.bandit_loggers.epsilon`

**Cause and message:**

<p>epsilon must lie in (0, 1], got {epsilon!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `epsilon`

**Registering modules:** `increment.simulate.bandit_loggers`

<span id="simulate.bandit_loggers.reward_mean_out_of_range"></span>

### `simulate.bandit_loggers.reward_mean_out_of_range`

**Cause and message:**

<p>reward mean for action={action!r}, x={x} is {mean!r}, outside [0, 1]; Bernoulli rewards need every cell mean inside the unit interval</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `action`, `mean`, `x`

**Registering modules:** `increment.simulate.bandit_loggers`

<span id="simulate.cluster_dgp.invalid_scenario"></span>

### `simulate.cluster_dgp.invalid_scenario`

**Cause and message:**

<p>invalid clustered CATE request: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment.simulate._cluster_reconstruction`, `increment.simulate.cluster_dgp`

<span id="simulate.cluster_dgp.overlap_roster_unavailable"></span>

### `simulate.cluster_dgp.overlap_roster_unavailable`

**Cause and message:**

<p>The public overlap-subpopulation result does not expose its retained evaluation roster</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.simulate.cluster_dgp`

<span id="simulate.dgp.binomial_fixture_counts"></span>

### `simulate.dgp.binomial_fixture_counts`

**Cause and message:**

<p>binomial fixture counts out of range: need 0 &lt;= x_c &lt;= n_c, 0 &lt;= x_t &lt;= n_t, n_c &gt;= 1, n_t &gt;= 1; got x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `n_c`, `n_t`, `x_c`, `x_t`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.conversion_lift_unattainable"></span>

### `simulate.dgp.conversion_lift_unattainable`

**Cause and message:**

<p>conversion lift {lift} is unattainable: it requires a marginal rate of {target:.4f} &gt;= 1 (control rate {p_control:.4f})</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `lift`, `p_control`, `target`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.count_rate_unrepresentable"></span>

### `simulate.dgp.count_rate_unrepresentable`

**Cause and message:**

<p>count rate cannot be represented by the numeric generator ({reason}). {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`, `route`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.scenario.true_lift_finite"></span>

### `simulate.dgp.scenario.true_lift_finite`

**Cause and message:**

<p>true_lift[{metric!r}] must be finite and &gt;= -0.99; got {value!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric`, `value`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.switchback.extra_fields"></span>

### `simulate.dgp.switchback.extra_fields`

**Cause and message:**

<p>unsupported SwitchbackScenario fields: {fields!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `fields`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.switchback.retained_window"></span>

### `simulate.dgp.switchback.retained_window`

**Cause and message:**

<p>declared_carryover_order must be less than observation_steps to retain observations; got {declared_carryover_order} and {observation_steps}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared_carryover_order`, `observation_steps`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.unknown_metric_true"></span>

### `simulate.dgp.unknown_metric_true`

**Cause and message:**

<p>Unknown metric &#x27;{metric_name}&#x27; in true_lift</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_name`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.unsupported_metric_supported"></span>

### `simulate.dgp.unsupported_metric_supported`

**Cause and message:**

<p>Unsupported metric(s): {unknown}. Supported: {supported}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `supported`, `unknown`

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.dgp.uptake_compliance_t"></span>

### `simulate.dgp.uptake_compliance_t`

**Cause and message:**

<p>uptake_compliance_t and tau_complier must be set together to generate an encouragement-design uptake + outcome DGP</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.simulate.dgp`

<span id="simulate.runner.cate_routing_does"></span>

### `simulate.runner.cate_routing_does`

**Cause and message:**

<p>CATE routing does not support metric {metric_name!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `metric_name`

**Registering modules:** `increment.simulate.runner`

<span id="simulate.runner.replications_invalid"></span>

### `simulate.runner.replications_invalid`

**Cause and message:**

<p>replications must be a positive integer, got {replications!r}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `replications`

**Registering modules:** `increment.simulate.runner`

<span id="family-source"></span>

## `source.*`

<span id="source.breakout.assignment_counts"></span>

### `source.breakout.assignment_counts`

**Cause and message:**

<p>unit_counts() is not supported on a breakout-scoped moments source; its rows are per (group_id, dimension_value), so a per-group count would silently report one segment&#x27;s n as the whole arm&#x27;s. Use DefinitionsMomentSource.moments_source() for total-grain counts.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment.sources`

<span id="source.breakout.compliance_summary"></span>

### `source.breakout.compliance_summary`

**Cause and message:**

<p>compliance_summary() is not supported on a breakout-scoped moments source; design-level compliance is a whole-population cohort, never conditioned on a breakout dimension. Read it off the unscoped MomentSource.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment.sources`

<span id="source.compliance_arm.cluster_family_declaration_mismatch"></span>

### `source.compliance_arm.cluster_family_declaration_mismatch`

**Cause and message:**

<p>ComplianceArm for group {group_id!r}: n_clusters={n_clusters!r} but cluster-uptake bivariate family present={has_family}; both must agree</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `has_family`, `n_clusters`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_arm.cluster_family_finite"></span>

### `source.compliance_arm.cluster_family_finite`

**Cause and message:**

<p>ComplianceArm for group {group_id!r}: {name}={value!r} is not finite</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `name`, `value`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_arm.n_clusters_positive"></span>

### `source.compliance_arm.n_clusters_positive`

**Cause and message:**

<p>ComplianceArm for group {group_id!r} needs n_clusters &gt;= 1, got {n_clusters}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `n_clusters`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_arm.n_units_positive"></span>

### `source.compliance_arm.n_units_positive`

**Cause and message:**

<p>ComplianceArm for group {group_id!r} needs n_units &gt;= 1, got {n_units}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `n_units`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_arm.partial_cluster_family"></span>

### `source.compliance_arm.partial_cluster_family`

**Cause and message:**

<p>ComplianceArm for group {group_id!r} carries a partial cluster-uptake bivariate family; ref_uptake/cluster_uptake1/cluster_uptake2/ref_size/cluster_size1/cluster_size2/cluster_cross must all be present together or all be None</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_arm.uptake_total_bounds"></span>

### `source.compliance_arm.uptake_total_bounds`

**Cause and message:**

<p>ComplianceArm for group {group_id!r}: uptake_total={uptake_total!r} must be finite and within [0, n_units={n_units}]</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `group_id`, `n_units`, `uptake_total`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.cluster_family_mismatch"></span>

### `source.compliance_summary.cluster_family_mismatch`

**Cause and message:**

<p>ComplianceSummary for study {study_id!r}: declared cluster={declared_cluster!r} but arm {group_id!r} carries cluster-uptake family present={has_family}; every arm must match the summary&#x27;s own declared cluster grain</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `declared_cluster`, `group_id`, `has_family`, `study_id`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.completed_window_required"></span>

### `source.compliance_summary.completed_window_required`

**Cause and message:**

<p>completed compliance requires an as-of day and a bounded uptake window</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.design_mismatch"></span>

### `source.compliance_summary.design_mismatch`

**Cause and message:**

<p>ComplianceSummary for study {study_id!r} was built for cohort={cohort!r} window_days={window_days!r} one_sided={one_sided!r}, but the requested design declares uptake.fact={design_fact!r} window_days={design_window_days!r} one_sided={design_one_sided!r}; control={control_group!r} versus {design_control_group!r}; these must agree before a caller can trust it</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cohort`, `control_group`, `design_control_group`, `design_fact`, `design_one_sided`, `design_window_days`, `one_sided`, `study_id`, `window_days`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.duplicate_arms"></span>

### `source.compliance_summary.duplicate_arms`

**Cause and message:**

<p>ComplianceSummary for study {study_id!r} carries duplicate arm(s) {duplicates!r}; exactly one canonical arm per group_id is required</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `duplicates`, `study_id`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.invalid_state"></span>

### `source.compliance_summary.invalid_state`

**Cause and message:**

<p>Invalid compliance sufficient state: {reason}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._evidence_types`

<span id="source.compliance_summary.legacy_uptake_state"></span>

### `source.compliance_summary.legacy_uptake_state`

**Cause and message:**

<p>ComplianceSummary for study {study_id!r} is unavailable from this reloaded source: {reason}; re-export moments/artifacts built after compliance_summary support was added</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `reason`, `study_id`

**Registering modules:** `increment._evidence_types`

<span id="source.frame.cluster_capability"></span>

### `source.frame.cluster_capability`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L80).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `feature`, `mechanism`, `metrics`, `route`, `uptake`

**Registering modules:** `increment._frame_validation`, `increment.frame`

<span id="source.frame.cluster_labels"></span>

### `source.frame.cluster_labels`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L143).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `cluster`, `examples`, `missing`, `reason`, `route`, `spanning`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.column_dtype"></span>

### `source.frame.column_dtype`

**Cause and message:**

<p>column {column!r} has dtype {dtype} which increment cannot interpret as a numeric metric value on every backend by construction; offending value(s): {offending_values}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `dtype`, `offending_values`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.compliance_asof"></span>

### `source.frame.compliance_asof`

**Cause and message:**

<p>{operation} cannot serve as_of={as_of!r} from a totals-shape frame without a day axis. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `as_of`, `operation`, `route`

**Registering modules:** `increment.frame`

<span id="source.frame.compliance_design_mismatch"></span>

### `source.frame.compliance_design_mismatch`

**Cause and message:**

<p>requested design {requested_design!r} differs from source design {source_design!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `requested_design`, `route`, `source_design`

**Registering modules:** `increment.frame`

<span id="source.frame.compliance_no_uptake"></span>

### `source.frame.compliance_no_uptake`

**Cause and message:**

<p>{source} has no declared encouragement uptake column. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `route`, `source`

**Registering modules:** `increment.frame`

<span id="source.frame.constructor"></span>

### `source.frame.constructor`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L73).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `capability`, `metric`, `route`, `value`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.control_invalid"></span>

### `source.frame.control_invalid`

**Cause and message:**

<p>control must be a scalar group label; received {value!r}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `value`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.control_missing"></span>

### `source.frame.control_missing`

**Cause and message:**

<p>control group {control!r} not found in column {group!r}; observed groups: {observed!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `control`, `group`, `observed`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.conversion_not_binary"></span>

### `source.frame.conversion_not_binary`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L135).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `metric`, `route`, `values`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.duplicate_columns"></span>

### `source.frame.duplicate_columns`

**Cause and message:**

<p>the frame repeats column name(s) {columns!r}; rename or drop the repeated columns</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.duplicate_units"></span>

### `source.frame.duplicate_units`

**Cause and message:**

<p>from_unit_summary requires one row per unit, but {count} unit(s) have duplicates; sample unit ids: {units!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `route`, `units`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.input_type"></span>

### `source.frame.input_type`

**Cause and message:**

<p>{constructor} does not accept {input_type} input; use {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructor`, `input_type`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.intervention_grain_without_cluster"></span>

### `source.frame.intervention_grain_without_cluster`

**Cause and message:**

<p>intervention_grain={intervention_grain!r} declares a whole-cluster policy deployment, but no cluster= column was declared; there is no cluster grain to intervene on.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `intervention_grain`

**Registering modules:** `increment.frame`

<span id="source.frame.metric_missing"></span>

### `source.frame.metric_missing`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L98).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `available`, `message`, `missing`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.non_finite"></span>

### `source.frame.non_finite`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L171).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `count`, `role`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.null_identity"></span>

### `source.frame.null_identity`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L164).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `count`, `role`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.on_unassigned_invalid"></span>

### `source.frame.on_unassigned_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L221).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `allowed`, `value`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.quantile_daily"></span>

### `source.frame.quantile_daily`

**Cause and message:**

<p>quantile metric {metric!r} does not decompose into per-day moments. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment.frame`

<span id="source.frame.quantile_no_moments"></span>

### `source.frame.quantile_no_moments`

**Cause and message:**

<p>quantile metric {metric!r} has no moment representation; it is served through unit_frame. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment.frame`, `increment.sources`

<span id="source.frame.ratio_same_column"></span>

### `source.frame.ratio_same_column`

**Cause and message:**

<p>metric {metric!r} reads column {column!r} for both numerator and denominator; that ratio is identically 1, so there is nothing to estimate; point the two sides at different columns</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `metric`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.retention_daily"></span>

### `source.frame.retention_daily`

**Cause and message:**

<p>retention metric {metric!r} has no independent per-day reading. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment.frame`

<span id="source.frame.sequential_exposure_date"></span>

### `source.frame.sequential_exposure_date`

**Cause and message:**

<p>{constructor}: sequential inference reveals units in exposure order, so it needs each unit&#x27;s exposure date. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructor`, `route`

**Registering modules:** `increment.frame`

<span id="source.frame.switchback.assignment"></span>

### `source.frame.switchback.assignment`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.columns"></span>

### `source.frame.switchback.columns`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.domain"></span>

### `source.frame.switchback.domain`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.identification"></span>

### `source.frame.switchback.identification`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_study.py#L36).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment._study`, `increment.switchback`

<span id="source.frame.switchback.metric"></span>

### `source.frame.switchback.metric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.missingness"></span>

### `source.frame.switchback.missingness`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.numeric"></span>

### `source.frame.switchback.numeric`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.plan"></span>

### `source.frame.switchback.plan`

**Cause and message:**

<p>switchback decision rejected for {reason!r}: {message}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `message`, `reason`

**Registering modules:** `increment.decision`

<span id="source.frame.switchback.schedule"></span>

### `source.frame.switchback.schedule`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.switchback.units"></span>

### `source.frame.switchback.units`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/switchback.py#L75).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.switchback`

<span id="source.frame.unassigned"></span>

### `source.frame.unassigned`

**Cause and message:**

<p>{count} row(s) have a null value in group column {group!r}; sample unit ids: {units!r}. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `constructor`, `count`, `group`, `route`, `units`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.unit_frame_panel"></span>

### `source.frame.unit_frame_panel`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L343).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`, `shape`

**Registering modules:** `increment._frame_validation`, `increment.frame`, `increment.query.artifact_reader`

<span id="source.frame.uptake_not_binary"></span>

### `source.frame.uptake_not_binary`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L127).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `route`, `uptake`, `values`

**Registering modules:** `increment._frame_validation`

<span id="source.frame.windowed_encouragement_total"></span>

### `source.frame.windowed_encouragement_total`

**Cause and message:**

<p>total moments for windowed metric {metric!r} under encouragement are unavailable from a unit panel. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `metric`, `route`

**Registering modules:** `increment.frame`

<span id="source.moments.assignment_counts"></span>

### `source.moments.assignment_counts`

**Cause and message:**

<p>trustworthy source-level assignment counts; use workflows.export_moments() from a source or artifact that carries assignment-count evidence, or supply a cube whose rows carry the assignment_counts field.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** —

**Registering modules:** `increment.sources`

<span id="source.native.compliance_design_mismatch"></span>

### `source.native.compliance_design_mismatch`

**Cause and message:**

<p>compliance_summary design mismatch: expected source design {expected!r}, received {received!r}; rebuild the source for the requested design</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `expected`, `received`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.covariate_ambiguous"></span>

### `source.native.covariate_ambiguous`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/_native_refusals.py#L35).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `candidates`, `covariate`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.covariate_as_of"></span>

### `source.native.covariate_as_of`

**Cause and message:**

<p>unit_frame: covariate {covariate!r} (source {source!r}) has as_of={as_of!r}; conditioning on a value measured at/after exposure biases the adjustment. Declare as_of=&#x27;pre_exposure&#x27; or &#x27;static&#x27; on the property.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `as_of`, `covariate`, `source`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.covariate_dtype"></span>

### `source.native.covariate_dtype`

**Cause and message:**

<p>unit_frame: covariate {covariate!r} (source {source!r}) has dtype {dtype!r}; covariate adjustment needs a numeric (int/float/bool) or categorical (string) column; a date carries no adjustment meaning. Derive a numeric or categorical pre-exposure property.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `covariate`, `dtype`, `source`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.covariate_kind_mismatch"></span>

### `source.native.covariate_kind_mismatch`

**Cause and message:**

<p>unit_frame: covariate {covariate!r} (source {source!r}) has property type {dtype!r}, incompatible with declared kind {kind!r}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `covariate`, `dtype`, `kind`, `source`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.covariate_reserved"></span>

### `source.native.covariate_reserved`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/native_contract.py#L174).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `cluster`, `covariate`

**Registering modules:** `increment.query.native_contract`, `increment.query.native_source`, `increment.query.source`

<span id="source.native.covariate_unresolved"></span>

### `source.native.covariate_unresolved`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/_native_refusals.py#L25).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `covariate`, `source`, `unit`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.experiment_key_unmatched"></span>

### `source.native.experiment_key_unmatched`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/_native_refusals.py#L82).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `experiment`, `observed_keys`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.native.fact_null_values"></span>

### `source.native.fact_null_values`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/query/_native_refusals.py#L100).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `fact`, `null_rows`

**Registering modules:** `increment.query._native_refusals`, `increment.query.fact_resolution`

<span id="source.native.trigger_evidence_required"></span>

### `source.native.trigger_evidence_required`

**Cause and message:**

<p>{operation} requires explicit SourceSnapshotEvidence from the upstream source; provide its event-time observation_cutoff_ts. Missing per-feed completeness certification remains unknown. {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `operation`, `route`

**Registering modules:** `increment.query._native_refusals`, `increment.query.native_source`

<span id="source.panel.as_of"></span>

### `source.panel.as_of`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_panel.py#L162).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `axis_kind`, `issue`, `value`

**Registering modules:** `increment._frame_panel`

<span id="source.panel.day_grain_columns"></span>

### `source.panel.day_grain_columns`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L156).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.panel.duplicate_unit_days"></span>

### `source.panel.duplicate_unit_days`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L200).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `date`, `examples`, `route`, `unit`

**Registering modules:** `increment._frame_validation`

<span id="source.panel.exposure_date"></span>

### `source.panel.exposure_date`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L192).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `count`, `examples`, `issue`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.panel.multi_group_unit"></span>

### `source.panel.multi_group_unit`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L208).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `count`, `examples`, `group`, `route`, `unit`

**Registering modules:** `increment._frame_validation`

<span id="source.panel.null_exposure"></span>

### `source.panel.null_exposure`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_validation.py#L184).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `column`, `constructor`, `count`, `route`

**Registering modules:** `increment._frame_validation`

<span id="source.panel.observation_end"></span>

### `source.panel.observation_end`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_frame_panel.py#L151).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `axis_kind`, `issue`, `value`

**Registering modules:** `increment._frame_panel`

<span id="family-store"></span>

## `store.*`

<span id="store.cell.not_found"></span>

### `store.cell.not_found`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.event_invalid"></span>

### `store.decision.event_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.log_corrupt"></span>

### `store.decision.log_corrupt`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.outcome_mismatch"></span>

### `store.decision.outcome_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.reference_invalid"></span>

### `store.decision.reference_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.schema_unsupported"></span>

### `store.decision.schema_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.decision.timestamp_invalid"></span>

### `store.decision.timestamp_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.history.ambiguous"></span>

### `store.history.ambiguous`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.corrupt"></span>

### `store.run.corrupt`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.duplicate"></span>

### `store.run.duplicate`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.empty"></span>

### `store.run.empty`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.identity_incomplete"></span>

### `store.run.identity_incomplete`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.identity_mismatch"></span>

### `store.run.identity_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.legacy_analysis_required"></span>

### `store.run.legacy_analysis_required`

**Cause and message:**

<p>Switchback results do not have a registered family decision plan; use {route}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `route`

**Registering modules:** `increment._source_workflows`

<span id="store.run.manifest_unsupported"></span>

### `store.run.manifest_unsupported`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L41).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `found`, `supported`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.metadata_mismatch"></span>

### `store.run.metadata_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.not_found"></span>

### `store.run.not_found`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.provenance_mismatch"></span>

### `store.run.provenance_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.roster_mismatch"></span>

### `store.run.roster_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.schema_unavailable"></span>

### `store.run.schema_unavailable`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L53).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `CapabilityError`

**Context keys:** `schema`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.scope_manifest_invalid"></span>

### `store.run.scope_manifest_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.scope_mapping_invalid"></span>

### `store.run.scope_mapping_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.run.timestamp_invalid"></span>

### `store.run.timestamp_invalid`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/kernel/program_refusals.py#L9).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `message`

**Registering modules:** `increment.kernel.program_refusals`

<span id="store.schema_name_too_long"></span>

### `store.schema_name_too_long`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_warehouse_namespace.py#L22).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `backend`, `limit`, `schema`, `store`

**Registering modules:** `increment._warehouse_namespace`

<span id="family-tables"></span>

## `tables.*`

<span id="tables.readout_table_ambiguous"></span>

### `tables.readout_table_ambiguous`

**Cause and message:**

<p>readout_table has ambiguous duplicate row keys {key}; {route}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `key`, `route`

**Registering modules:** `increment.tables`

<span id="tables.readout_table_duplicate"></span>

### `tables.readout_table_duplicate`

**Cause and message:**

<p>readout_table has duplicate visible row keys after disambiguation; include distinct metric, estimand, value_scale, nest, split, or analysis_population values.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.tables`

<span id="tables.readout_table_nest"></span>

### `tables.readout_table_nest`

**Cause and message:**

<p>readout_table(nest_by=&#x27;segment&#x27;) needs a single method in data; filter the &#x27;method&#x27; column to one value so segment and arm can each use a table axis.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.tables`

<span id="tables.readout_table_trend"></span>

### `tables.readout_table_trend`

**Cause and message:**

<p>readout_table(trend=...) is missing the &#x27;ds&#x27; column needed for the sparkline&#x27;s x-axis; got columns {columns}.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `columns`

**Registering modules:** `increment.tables`

<span id="tables.readout_table_trend_ds_all_null"></span>

### `tables.readout_table_trend_ds_all_null`

**Cause and message:**

<p>readout_table(trend=...): every row&#x27;s &#x27;ds&#x27; is null; these estimates carry no per-date axis (a whole-window run() result?). Build the trend from readouts.asof_lift or run_daily_lift output, whose rows each carry their own day.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** —

**Registering modules:** `increment.tables`

<span id="tables.readout_table_trend_duplicate_rows"></span>

### `tables.readout_table_trend_duplicate_rows`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/tables.py#L103).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `key`, `key_columns`, `rows`, `varying`

**Registering modules:** `increment.tables`

<span id="tables.readout_table_trend_missing_key_columns"></span>

### `tables.readout_table_trend_missing_key_columns`

**Cause and message:**

<p>readout_table(trend=...) is missing column(s) {missing} required by nest_by={nest_by!r} to align each trend row with the main table&#x27;s metric/nest/split keys ({key_columns}).</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `key_columns`, `missing`, `nest_by`

**Registering modules:** `increment.tables`

<span id="tables.readout_table_trend_mixed_ds_basis"></span>

### `tables.readout_table_trend_mixed_ds_basis`

**Cause and message:**

<p>readout_table(trend=...) mixes ds_basis values {bases}{affected}: a cohort-indexed series (ds is each unit&#x27;s own exposure date) cannot share an x-axis with a calendar-indexed series. Plot the cohort view separately, or use the metric&#x27;s as-of view (run_asof/run_asof_lift), which is calendar-indexed.</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `affected`, `bases`

**Registering modules:** `increment.tables`

<span id="tables.trend_table_missing"></span>

### `tables.trend_table_missing`

**Cause and message:**

<p>trend_table is missing required column(s) {missing}; pass MetricTrend.to_frame() output</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `missing`

**Registering modules:** `increment.tables`

<span id="tables.trend_table_supports"></span>

### `tables.trend_table_supports`

**Cause and message:**

<p>trend_table supports at most one dimension column; got {dims}</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `dims`

**Registering modules:** `increment.tables`

<span id="family-unit_cycle"></span>

## `unit_cycle.*`

<span id="unit_cycle.envelope_required"></span>

### `unit_cycle.envelope_required`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.error_budget_exhausted"></span>

### `unit_cycle.error_budget_exhausted`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.implausible_realized_split"></span>

### `unit_cycle.implausible_realized_split`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.invalid_design"></span>

### `unit_cycle.invalid_design`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.numerical"></span>

### `unit_cycle.numerical`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.reference_mismatch"></span>

### `unit_cycle.reference_mismatch`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="unit_cycle.sufficient_state"></span>

### `unit_cycle.sufficient_state`

**Cause and message:**

The cause depends on the supplied context; read the raised message. [Message rule](https://github.com/kylejcaron/increment/blob/main/increment/_unit_cycle.py#L29).

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `InvalidRequestError`

**Context keys:** `reason`

**Registering modules:** `increment._unit_cycle`

<span id="family-wire"></span>

## `wire.*`

<span id="wire.procedure.contrast_context"></span>

### `wire.procedure.contrast_context`

**Cause and message:**

<p>switchback contrast runs carry no portable arm decision plan; {action} is unavailable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `action`

**Registering modules:** `increment._run`

<span id="wire.result.raw_outcomes_unavailable"></span>

### `wire.result.raw_outcomes_unavailable`

**Cause and message:**

<p>a live confidence-set result depends on raw outcomes; {action} is unavailable</p>

**Action:** Check the named inputs against the message before retrying. [Recovery guidance](../guides/refusals.md#find-the-next-step-by-code-family).

**Error class:** `WireFormatError`

**Context keys:** `action`

**Registering modules:** `increment._run`


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
| `frame.missing_policy.panel_drop` | `frame.missing_policy.panel_skip` |
| `frame.moments.metric_covariate_column` | `frame.validation.metric_covariate_impute_all_null` |
| `frame.panel.observable_end_dtype` | `source.panel.observation_end` |
| `frame.validation.metric_missing_drop` | Removed without replacement |
| `kernel.resolver.capabilities_missing` | `kernel.provider.capabilities_missing` |
| `kernel.resolver.capability_mismatch` | `kernel.provider.capability_mismatch` |
| `kernel.resolver.digest_mismatch` | `kernel.provider.digest_mismatch` |
| `kernel.resolver.provider_mismatch` | `kernel.provider.mismatch` |
| `kernel.run.provider_execution_unavailable` | `kernel.run.reader_required` |
| `kernel.run.resolver_missing` | `kernel.run.reader_required`, `kernel.run.reader_offline` |
| `plan.contrast_configs_template_mismatch` | `plan.compile_configs_template_mismatch` |
| `plan.contrast_configs_wrong_type` | `plan.compile_configs_wrong_type` |
| `plan.metric_informative_priors` | `sequential.route.unsupported` |
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
| `readout.estimate_lift_every` | Removed without replacement |
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
