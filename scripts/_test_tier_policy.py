"""Canonical pytest mark expressions for supported test tiers."""

TIER_MARKERS = {
    "fast": "not slow and not parameter_recovery and not numerical_certification",
    "slow": ("slow and not parameter_recovery and not numerical_certification and not examples"),
    "scientific": "(parameter_recovery or numerical_certification)",
    "numerical-certification": "numerical_certification",
}


def tier_mark_expression(tier: str, additional: str | None = None) -> str:
    expression = TIER_MARKERS[tier]
    return f"({expression}) and ({additional})" if additional else expression
