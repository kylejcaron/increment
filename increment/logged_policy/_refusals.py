"""Coded refusals owned by the logged-policy evidence family."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)


def _where(unit_id: object, decision_index: object) -> str:
    return f"decision (unit_id={unit_id!r}, decision_index={decision_index!r})"


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "logged_policy.trace.propensity_missing": RefusalSpec(
            "logged_policy.trace.propensity_missing",
            InvalidRequestError,
            lambda *, unit_id, decision_index, logging_policy_id, logging_policy_version: (
                f"{_where(unit_id, decision_index)} under "
                f"{logging_policy_id}/{logging_policy_version} has no logged propensity; the exact "
                "logging probability is evidence and is never inferred from the registered policy"
            ),
        ),
        "logged_policy.trace.propensity_nonfinite": RefusalSpec(
            "logged_policy.trace.propensity_nonfinite",
            InvalidRequestError,
            lambda *, unit_id, decision_index, propensity: (
                f"{_where(unit_id, decision_index)} has a non-finite propensity {propensity!r}"
            ),
        ),
        "logged_policy.trace.propensity_out_of_range": RefusalSpec(
            "logged_policy.trace.propensity_out_of_range",
            InvalidRequestError,
            lambda *, unit_id, decision_index, propensity: (
                f"{_where(unit_id, decision_index)} has propensity {propensity!r} outside (0, 1)"
            ),
        ),
        "logged_policy.trace.propensity_below_floor": RefusalSpec(
            "logged_policy.trace.propensity_below_floor",
            InvalidRequestError,
            lambda *, unit_id, decision_index, propensity, floor, action=None, logging_policy_id=None, logging_policy_version=None, context=None, route=None: (
                f"{_where(unit_id, decision_index)} has propensity {propensity!r} for "
                f"action {action!r} under {logging_policy_id}/{logging_policy_version} "
                f"below the positivity floor {floor}; the trace is refused rather than "
                f"clipped or trimmed. {route}"
            ),
        ),
        "logged_policy.trace.propensity_stale": RefusalSpec(
            "logged_policy.trace.propensity_stale",
            InvalidRequestError,
            lambda *, unit_id, decision_index, logging_policy_id, logging_policy_version, chosen_action, logged, registered: (
                f"{_where(unit_id, decision_index)} logs propensity {logged!r} for action "
                f"{chosen_action!r}, but the supplied {logging_policy_id}/{logging_policy_version} "
                f"distribution assigns {registered!r} at that history; the logged value is stale"
            ),
        ),
        "logged_policy.trace.logging_policy_unregistered": RefusalSpec(
            "logged_policy.trace.logging_policy_unregistered",
            InvalidRequestError,
            lambda *, unit_id, decision_index, logging_policy_id, logging_policy_version, registered: (
                f"{_where(unit_id, decision_index)} was logged by "
                f"{logging_policy_id}/{logging_policy_version}, which is not in the policy registry "
                f"{registered!r}; the full logging distribution must be reproducible"
            ),
        ),
        "logged_policy.trace.registry_audit_required": RefusalSpec(
            "logged_policy.trace.registry_audit_required",
            InvalidRequestError,
            lambda *, constructors: (
                f"LoggedTrace must be built through {' or '.join(constructors)} so every logging "
                "distribution is admitted through a supported route"
            ),
        ),
        "logged_policy.trace.admission_route": RefusalSpec(
            "logged_policy.trace.admission_route",
            InvalidRequestError,
            lambda *, supplied, alternatives: (
                f"exactly one of {alternatives!r} must be supplied; received {supplied!r}"
            ),
        ),
        "logged_policy.trace.context_value_unsupported": "pre-decision context contains unsupported value type {value_type!r}; use nested mappings, sequences, sets, or supported immutable scalar values",
        "logged_policy.trace.open_reward": RefusalSpec(
            "logged_policy.trace.open_reward",
            InvalidRequestError,
            lambda *, unit_id, decision_index, reward_observation_boundary: (
                f"{_where(unit_id, decision_index)} has no closed reward at boundary "
                f"{reward_observation_boundary!r}; an open reward is never treated as zero"
            ),
        ),
        "logged_policy.trace.reward_nonfinite": RefusalSpec(
            "logged_policy.trace.reward_nonfinite",
            InvalidRequestError,
            lambda *, unit_id, decision_index, reward: (
                f"{_where(unit_id, decision_index)} has a non-finite reward {reward!r}"
            ),
        ),
        "logged_policy.trace.unordered": "unit {unit_id!r} decisions are not a consecutive 1..T sequence with strictly increasing decision_time: indices={decision_indices!r}, times={decision_times!r}",
        "logged_policy.trace.incomplete_horizon": "unit {unit_id!r} has {observed_horizon} closed decisions but the fixed horizon is {horizon}; every admitted unit needs all {horizon} rewards closed",
        "logged_policy.trace.candidate_set_mismatch": RefusalSpec(
            "logged_policy.trace.candidate_set_mismatch",
            InvalidRequestError,
            lambda *, unit_id, decision_index, candidate_actions, expected_candidate_actions, chosen_action: (
                f"{_where(unit_id, decision_index)} declares candidate_actions={candidate_actions!r} "
                f"with chosen_action={chosen_action!r}; the trace requires the fixed ordered set "
                f"{expected_candidate_actions!r} containing the chosen action"
            ),
        ),
        "logged_policy.trace.logging_distribution_keys": RefusalSpec(
            "logged_policy.trace.logging_distribution_keys",
            InvalidRequestError,
            lambda *, unit_id, decision_index, actions, expected_actions: (
                f"{_where(unit_id, decision_index)} records logging distribution keys "
                f"{actions!r}, expected exactly {expected_actions!r}"
            ),
        ),
        "logged_policy.trace.empty": RefusalSpec(
            "logged_policy.trace.empty",
            InvalidRequestError,
            lambda *, n_records, constructors: (
                f"a logged trace needs at least one decision record, got {n_records}; pass the "
                f"records to {' or '.join(constructors)}"
            ),
        ),
        "logged_policy.trace.boundary_before_decision": RefusalSpec(
            "logged_policy.trace.boundary_before_decision",
            InvalidRequestError,
            lambda *, unit_id, decision_index, decision_time, reward_observation_boundary: (
                f"{_where(unit_id, decision_index)} closes its reward at "
                f"{reward_observation_boundary!r}, before the decision at {decision_time!r}"
            ),
        ),
        "logged_policy.trace.boundary_awareness_mixed": RefusalSpec(
            "logged_policy.trace.boundary_awareness_mixed",
            InvalidRequestError,
            lambda *, unit_id, decision_index, decision_time, reward_observation_boundary: (
                f"{_where(unit_id, decision_index)} mixes a naive and a timezone-aware timestamp: "
                f"decision_time={decision_time!r}, reward_observation_boundary="
                f"{reward_observation_boundary!r}; both must be naive or both timezone-aware"
            ),
        ),
        "logged_policy.trace.decision_time_awareness_mixed": RefusalSpec(
            "logged_policy.trace.decision_time_awareness_mixed",
            InvalidRequestError,
            lambda *, unit_id, decision_index, decision_time, reference_unit_id, reference_decision_index, reference_decision_time: (
                f"{_where(unit_id, decision_index)} has decision_time {decision_time!r} but "
                f"{_where(reference_unit_id, reference_decision_index)} has {reference_decision_time!r}; "
                "every decision_time in a trace must be naive or every one timezone-aware"
            ),
        ),
        "logged_policy.trace.missing_columns": "trace frame is missing required columns {missing!r}",
        "logged_policy.trace.unit_id_missing": "trace frame row {row} has no unit_id ({unit_id!r}); every decision must belong to an identifiable unit",
        "logged_policy.trace.logging_distribution_misaligned": "{n_records} records but {n_distributions} supplied logging distributions; build a trace with LoggedTrace.from_records or from_frame",
        "logged_policy.registry.duplicate": "policy {policy_id}/{version} is registered twice; a version is immutable",
        "logged_policy.support.context_unsupported": RefusalSpec(
            "logged_policy.support.context_unsupported",
            CapabilityError,
            lambda *, policy_id, version, context_key, context: (
                f"policy {policy_id}/{version} has no distribution for {context_key}="
                f"{context.get(context_key)!r} in context {context!r}"
            ),
        ),
        "logged_policy.support.policy_not_normalized": RefusalSpec(
            "logged_policy.support.policy_not_normalized",
            CapabilityError,
            template="policy {policy_id}/{version} assigns {probabilities!r} over the candidate set at context {context!r}; probabilities must lie in [0, 1] and sum to 1",
        ),
        "logged_policy.policy.json_context_key": RefusalSpec(
            "logged_policy.policy.json_context_key",
            CapabilityError,
            template="policy {policy_id}/{version} has context-table keys of type {key_types!r} that JSON cannot represent without changing the policy; {route}",
        ),
        "logged_policy.support.action_unsupported": RefusalSpec(
            "logged_policy.support.action_unsupported",
            CapabilityError,
            lambda *, unit_id, decision_index, policy_id, version, action, policy_probability: (
                f"{_where(unit_id, decision_index)}: evaluated policy {policy_id}/{version} gives "
                f"action {action!r} probability {policy_probability!r} but the logging policy gave it "
                "probability 0; positivity fails and the estimand is not identified"
            ),
        ),
        "logged_policy.support.ess_floor": RefusalSpec(
            "logged_policy.support.ess_floor",
            CapabilityError,
            template="effective sample size for {policy_id}/{version} at decision_index={decision_index} is {ess:.4f}, below the floor {floor}; cumulative weights are concentrated on fewer than {floor} effective units (target ESS by time {target_ess_by_time!r}, reference ESS by time {reference_ess_by_time!r})",
        ),
        "logged_policy.inference.adaptive_logging_unsupported": RefusalSpec(
            "logged_policy.inference.adaptive_logging_unsupported",
            CapabilityError,
            lambda *, logging_policy_versions, logging_policy_keys, update_batches: (
                f"the trace was logged under {len(logging_policy_keys)} policy versions "
                f"{logging_policy_versions!r} with structured keys {logging_policy_keys!r} "
                f"(update batches {update_batches!r}); the unit-clustered "
                "sandwich interval is only calibrated under one fixed logging law, and inference for "
                "adaptively collected data (stabilized or adaptively weighted estimators with a "
                "martingale variance, as in Hadad et al. 2021 and Zhang, Janson and Murphy 2021) is "
                "not implemented -- evaluate a trace logged by a single policy version"
            ),
        ),
        "logged_policy.inference.cluster_floor": RefusalSpec(
            "logged_policy.inference.cluster_floor",
            CapabilityError,
            template="unit-clustered sandwich inference needs at least {minimum_total_units} independent units, got {n_units} (ESS by time {ess_by_time!r}, max cumulative weight by time {max_weight_by_time!r}, min propensity by time {min_propensity_by_time!r})",
        ),
        "logged_policy.inference.alpha": "alpha must be a float in (0, 1), got {alpha!r}",
        "logged_policy.inference.unrepresentable": RefusalSpec(
            "logged_policy.inference.unrepresentable",
            CapabilityError,
            lambda *, nonfinite, estimate, se, target_value, reference_value: (
                f"the contrast has no float64 representation ({', '.join(nonfinite)} not finite): "
                f"estimate={estimate!r}, se={se!r}, target_value={target_value!r}, "
                f"reference_value={reference_value!r}"
            ),
        ),
    },
)
_raise = raiser(_REFUSALS)

LOGGED_POLICY_REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(_REFUSALS)
