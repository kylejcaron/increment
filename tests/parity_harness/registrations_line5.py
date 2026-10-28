from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from increment.errors import CodedError
from tests.parity_harness.cases import PARITY_CASES

_LINE5_SCENARIOS = (
    "exact_binomial_beyond_the_former_arm_ceiling",
    "sequential_automatic_multi_arm_exact",
    "sequential_boolean_segments_automatic_exact_bernoulli",
    "sequential_boolean_segments_registered_exact_bernoulli",
    "sequential_explicit_registration_native_artifact_identity",
    "sequential_registered_registration_native_artifact_identity",
    "audit-sequential-messy-mean-conversion",
    "audit-sequential-missing-zero",
    "scalar_sequential_fixed_threshold_winsor",
    "scalar_sequential_mean_cuped_asymptotic",
    "sequential-mixed-quantile-catalog-checkpoint",
    "sequential_asymptotic_mean_revenue",
    "sequential_boolean_segments_automatic_asymptotic_mean",
    "sequential_boolean_segments_registered_asymptotic_mean",
)
_LINE5_PAIRS = frozenset((scenario_id, "from_switchback_panel") for scenario_id in _LINE5_SCENARIOS)


def _switchback_outcome(scenario_id: str, ingress: str) -> Mapping[str, object]:
    case = next(case for case in PARITY_CASES if case.id == scenario_id)
    if ingress not in case.build:
        return {"kind": "absent", "reason": case.waive[ingress]}

    try:
        case.build[ingress]()
    except CodedError as error:
        return {
            "kind": "refusal",
            "code": error.code,
            "stage": "construct",
            "context": dict(error.context),
        }
    raise AssertionError(
        f"{scenario_id}/{ingress} unexpectedly constructs; it is not a successful line-5 pair"
    )


REGISTERED: Mapping[str, frozenset[tuple[str, str]]] = {"line5": _LINE5_PAIRS}
NEW_PATH_ADAPTERS: Mapping[tuple[str, str], Any] = dict.fromkeys(_LINE5_PAIRS, _switchback_outcome)
