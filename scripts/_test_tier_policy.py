"""Canonical pytest mark expressions for supported test tiers."""

TIER_MARKERS = {
    "fast": "not slow and not parameter_recovery",
    "slow": "slow and not parameter_recovery and not examples",
}


def tier_mark_expression(tier: str, additional: str | None = None) -> str:
    expression = TIER_MARKERS[tier]
    return f"({expression}) and ({additional})" if additional else expression
