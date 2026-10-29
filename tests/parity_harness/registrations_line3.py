from __future__ import annotations

from collections.abc import Mapping

_LINE3_SCENARIOS = (
    "audit-sequential-messy-mean-conversion",
    "audit-sequential-missing-zero",
    "exact_binomial_beyond_the_former_arm_ceiling",
    "scalar_sequential_fixed_threshold_winsor",
    "scalar_sequential_mean_cuped_asymptotic",
    "sequential-mixed-quantile-catalog-checkpoint",
    "sequential_asymptotic_mean_revenue",
    "sequential_automatic_multi_arm_exact",
    "sequential_boolean_segments_automatic_asymptotic_mean",
    "sequential_boolean_segments_automatic_exact_bernoulli",
    "sequential_boolean_segments_registered_asymptotic_mean",
    "sequential_boolean_segments_registered_exact_bernoulli",
    "sequential_explicit_registration_native_artifact_identity",
    "sequential_registered_registration_native_artifact_identity",
)
_LINE3_INGRESSES = (
    "from_definitions",
    "from_moments",
    "from_unit_panel",
    "from_unit_summary",
)
LINE3_PAIRS = frozenset(
    (scenario_id, ingress) for scenario_id in _LINE3_SCENARIOS for ingress in _LINE3_INGRESSES
)


def line3_actual_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    from tests.test_evidence_identity_parity import _line1_actual_outcome

    return _line1_actual_outcome(scenario_id, ingress, scoped=True)
