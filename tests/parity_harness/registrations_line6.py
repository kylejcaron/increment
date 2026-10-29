from __future__ import annotations

from collections.abc import Callable, Mapping

_LINE6_SCENARIOS = frozenset(
    {
        "conversion_route_observational_family",
        "observational_aipw_conversion_diagnostics",
        "observational_aipw_dml_covariate_adjusted",
        "observational_categorical_covariate_adjusted",
        "observational_categorical_covariate_missing_levels",
        "observational_default_with_unadjusted_sensitivity",
        "observational_iptw_conversion_diagnostics",
        "observational_iptw_covariate_adjusted_ate",
        "observational_multiplicity_primary_secondary_guardrail",
        "observational_quantile_refused_at_the_readout_seam",
    }
)
_LINE6_INGRESSES = (
    "from_definitions",
    "from_moments",
    "from_unit_panel",
    "from_unit_summary",
)
PAIRS = frozenset(
    (scenario_id, ingress) for scenario_id in _LINE6_SCENARIOS for ingress in _LINE6_INGRESSES
)


def _line6_public_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    from tests.test_evidence_identity_parity import _line1_actual_outcome

    return _line1_actual_outcome(scenario_id, ingress)


ADAPTERS: Mapping[tuple[str, str], Callable[[str, str], Mapping[str, object]]] = dict.fromkeys(
    PAIRS, _line6_public_outcome
)
