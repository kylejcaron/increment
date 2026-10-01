import pytest

from increment._plan_compatibility import (
    PlanFamilyCompatibilityRequest,
    PlanFamilyDecision,
    plan_family_compatibility,
)


def _request(**overrides):
    values = {
        "metric": "revenue",
        "metric_type": "mean",
        "role": "secondary",
        "inference": "fixed",
    }
    values.update(overrides)
    return PlanFamilyCompatibilityRequest(**values)


def test_fixed_secondary_participates_without_runtime_effect():
    decision = plan_family_compatibility(_request())
    assert decision.runtime_effect == "not_applicable"
    assert decision.participation == "participates"
    assert decision.warning is None


def test_always_valid_quantile_is_limited_excluded_and_warned():
    decision = plan_family_compatibility(
        _request(metric="latency", metric_type="quantile", inference="always_valid")
    )
    assert decision.runtime_effect == "limited"
    assert decision.participation == "excluded"
    assert decision.warning is not None


def test_always_valid_quantile_primary_still_warns_without_family_participation():
    decision = plan_family_compatibility(
        _request(
            metric="latency",
            metric_type="quantile",
            role="primary",
            inference="always_valid",
        )
    )
    assert decision.runtime_effect == "limited"
    assert decision.participation == "not_applicable"
    assert decision.warning is not None


def test_nonsecondary_role_has_no_family_participation():
    decision = plan_family_compatibility(_request(role="primary"))
    assert decision.runtime_effect == "not_applicable"
    assert decision.participation == "not_applicable"


@pytest.mark.parametrize(
    ("runtime_effect", "participation", "warning"),
    [
        ("not_applicable", "excluded", None),
        ("not_applicable", "participates", "material warning"),
        ("limited", "participates", None),
        ("limited", "not_applicable", None),
    ],
)
def test_decision_rejects_runtime_effect_that_hides_limitation(
    runtime_effect,
    participation,
    warning,
):
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        PlanFamilyDecision(
            runtime_effect=runtime_effect,
            participation=participation,
            warning=warning,
        )
    assert exc_info.value.code == "facade.plan_compatibility.plan_family.runtime_effect_limited"


def test_metric_type_accepts_forward_compatible_metric_type_string():
    """New metric discriminators do not require a duplicate enum."""
    request = PlanFamilyCompatibilityRequest(
        metric="m",
        metric_type="duration_bucketed",
        role="secondary",
        inference="fixed",
    )
    decision = plan_family_compatibility(request)
    assert decision.participation == "participates"
