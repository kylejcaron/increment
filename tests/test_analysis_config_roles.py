from __future__ import annotations

import copy
import pickle

import pytest

from increment._analysis_config import effective_methods, overlay_configs, resolve_configs
from increment.errors import InvalidRequestError
from increment.estimation.engine import Method
from increment.semantics.design import AdjustmentSet, Observational, Randomized
from increment.semantics.models import MeanMetric


def test_default_method_policy_survives_config_copying():
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue")
    config = resolve_configs(
        [metric],
        None,
        None,
        methods=None,
        prior=None,
        sensitivity_methods=(Method(name="unadjusted"),),
    )[0]
    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",)))
    for copied in (
        config,
        copy.copy(config),
        copy.deepcopy(config),
        pickle.loads(pickle.dumps(config)),
    ):
        assert [method.name for method in effective_methods(copied, design=design)] == [
            "iptw",
            "unadjusted",
        ]
        with pytest.raises(InvalidRequestError) as raised:
            effective_methods(copied, design=Randomized(control_group="control"))
        assert raised.value.code == "estimation.engine.method_names_unique"
        assert raised.value.context["duplicates"] == ("unadjusted",)


def test_sensitivity_override_preserves_default_when_legacy_methods_lose_precedence():
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue")
    base = resolve_configs([metric], None, None, methods=None, prior=None)
    overlaid = overlay_configs(
        [metric],
        base,
        methods=[Method(name="dml")],
        prior=None,
        sensitivity_methods=(Method(name="unadjusted"),),
    )[0]
    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("x",)))
    assert [method.name for method in effective_methods(overlaid, design=design)] == [
        "iptw",
        "unadjusted",
    ]


def test_explicit_duplicate_methods_are_refused_before_design_resolution():
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue")
    with pytest.raises(InvalidRequestError) as raised:
        resolve_configs(
            [metric],
            None,
            None,
            methods=None,
            prior=None,
            decision_method=Method(name="unadjusted"),
            sensitivity_methods=(Method(name="unadjusted"),),
        )
    assert raised.value.code == "estimation.engine.method_names_unique"
    assert raised.value.context["duplicates"] == ("unadjusted",)
