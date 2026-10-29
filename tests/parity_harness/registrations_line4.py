_LINE4_SCENARIOS = (
    "audit-encouragement-multi-metric-breakout",
    "conversion_route_encouragement_family",
    "encouragement_declared_definitions",
    "encouragement_fcr_reestimation",
    "encouragement_guardrail_late_only_request",
    "encouragement_guardrail_late_rows",
    "encouragement_itt_dense_takes_the_delta_method_route",
    "encouragement_itt_keeps_binomial_reference_kind",
    "encouragement_margin_guardrail",
    "encouragement_margin_guardrail_late_only",
    "encouragement_optional_exclusion",
    "encouragement_supplied_design_definitions",
    "sequential_composed_family_ebh",
    "sequential_encouragement_optional_exclusion",
    "triggered-sitewide-population-results",
    "triggered_daily",
    "triggered_daily_lift",
    "triggered_encouragement_compliance_asof",
    "triggered_sequential_asymptotic_mean",
    "sequential_composed_itt_and_uptake",
    "triggered_asof",
    "triggered_asof_lift",
)
_LINE4_INGRESSES = ("from_definitions", "from_moments", "from_unit_panel", "from_unit_summary")
LINE4_PAIRS = frozenset(
    (scenario, ingress) for scenario in _LINE4_SCENARIOS for ingress in _LINE4_INGRESSES
) | frozenset({("sequential_composed_family_ebh", "from_switchback_panel")})


def merge_line4(registered, adapters, actual_outcome):
    return (
        {**registered, "line4": LINE4_PAIRS},
        {**adapters, **dict.fromkeys(LINE4_PAIRS, actual_outcome)},
    )
