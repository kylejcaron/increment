from __future__ import annotations

import pytest

from increment.errors import CodedError, RefusalSpec
from tests.refusal_registry import walk_refusal_registries


def _owned_codes() -> set[str]:
    registrations = walk_refusal_registries()
    return {
        code
        for code, owners in registrations.items()
        if code.startswith("estimation.engine.")
        and any(
            owner.startswith("increment.estimation.engine.") and isinstance(spec, RefusalSpec)
            for owner, spec in owners
        )
    } | {
        code
        for code, owners in registrations.items()
        if any(
            owner.startswith("increment.compatibility.") and isinstance(spec, RefusalSpec)
            for owner, spec in owners
        )
    }


def test_every_owned_code_has_a_scenario_or_a_reason() -> None:
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable

    assert set(REFUSAL_SCENARIOS) == _owned_codes()
    for code, entry in REFUSAL_SCENARIOS.items():
        assert isinstance(entry, Unreachable) or callable(entry), code
        if isinstance(entry, Unreachable):
            assert entry.reason.strip(), code


@pytest.mark.integration
@pytest.mark.parametrize("code", sorted(_owned_codes()))
def test_refusal_scenarios_raise_their_code_from_the_owner(code: str) -> None:
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable

    scenario = REFUSAL_SCENARIOS[code]
    if isinstance(scenario, Unreachable):
        pytest.skip(scenario.reason)
    with pytest.raises(CodedError) as raised:
        scenario()
    assert raised.value.code == code


def test_ratio_refusals_are_publicly_reachable() -> None:
    from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS, Unreachable

    for code in (
        "estimation.engine.ratio.cuped",
        "estimation.engine.ratio.negative_variance",
    ):
        assert not isinstance(REFUSAL_SCENARIOS[code], Unreachable)
