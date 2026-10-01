from __future__ import annotations

import math
from collections.abc import MutableMapping
from operator import setitem
from typing import cast

import pytest

from increment.decision import (
    ArmHypothesisKey,
    DecisionComputation,
    HypothesisKey,
)
from increment.errors import InvalidRequestError
from increment.estimation.decision_types import (
    ArmHypothesisKey as EstArmHypothesisKey,
)
from increment.estimation.decision_types import (
    DecisionComputation as EstDecisionComputation,
)
from increment.estimation.decision_types import EValueEvidence, PValueEvidence


def test_decision_reexports_decision_types_identity():
    assert ArmHypothesisKey is EstArmHypothesisKey
    assert DecisionComputation is EstDecisionComputation


def test_hypothesis_key_union_includes_estimation_members():
    key = ArmHypothesisKey(metric="conversion", group_id="treatment", estimand="itt")
    assert isinstance(key, HypothesisKey)


def _key():
    return ArmHypothesisKey(metric="rev", group_id="treatment", estimand="itt")


class TestPValueEvidenceFinite:
    @pytest.mark.parametrize("bad_p", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
    def test_out_of_domain_p_value_refused(self, bad_p):
        with pytest.raises(InvalidRequestError) as exc_info:
            PValueEvidence(
                hypothesis=_key(), method="unadjusted", p_value=bad_p, reference="normal"
            )
        assert exc_info.value.code == "decision.p_value_evidence.finite_unit_interval"
        if math.isnan(bad_p):
            assert math.isnan(cast("float", exc_info.value.context["p_value"]))
        else:
            assert exc_info.value.context["p_value"] == bad_p
        with pytest.raises(TypeError):
            exc_info.value.context["p_value"] = 0.5  # ty: ignore  -- context is immutable

    @pytest.mark.parametrize("boundary_p", [0.0, 1.0, 0.5])
    def test_boundary_and_ordinary_p_values_accepted(self, boundary_p):
        evidence = PValueEvidence(
            hypothesis=_key(), method="unadjusted", p_value=boundary_p, reference="normal"
        )
        assert evidence.p_value == boundary_p


class TestEValueEvidenceCertified:
    @pytest.mark.slow
    @pytest.mark.parametrize("bad_log", [float("nan"), float("inf"), float("-inf"), -1.0])
    def test_changed_log_cannot_relabel_actual_likelihood(self, bad_log):
        from dataclasses import replace

        from increment import AlwaysValid, estimate_sequential
        from increment.errors import CapabilityError
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        reg = registration()
        bundle = estimate_sequential(
            capture(reg, records([0, 0, 0, 1] * 16, [0, 1, 1, 1] * 16)),
            AlwaysValid(registration=reg),
        )
        evidence = next(iter(bundle.evidence.values()))
        assert isinstance(evidence, EValueEvidence)
        assert evidence.log_e > 0
        with pytest.raises(CapabilityError) as raised:
            replace(evidence, log_e=bad_log)
        assert raised.value.code == "sequential.source.invalid"
        # Deliberately violate the read-only type to test runtime immutability.
        with pytest.raises(TypeError):
            setitem(cast(MutableMapping[str, object], raised.value.context), "reason", "changed")

    @pytest.mark.slow
    def test_actual_evidence_requires_its_verified_source_prefix(self):
        from increment import AlwaysValid, estimate_sequential
        from increment.errors import CapabilityError
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        reg = registration()
        bundle = estimate_sequential(
            capture(reg, records([0, 1] * 8, [1, 1] * 8)), AlwaysValid(registration=reg)
        )
        with pytest.raises(CapabilityError) as raised:
            DecisionComputation(results=bundle.results, evidence=bundle.evidence, failures={})
        assert raised.value.code == "sequential.source.invalid"
        copied = DecisionComputation(
            results=bundle.results,
            evidence=bundle.evidence,
            failures={},
            sequential_snapshot=bundle.sequential_snapshot,
        )
        assert copied.results[0].stat_sig() == bundle.results[0].stat_sig()
