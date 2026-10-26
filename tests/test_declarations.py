from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from increment.errors import CodedError, UnsupportedRequestError


def test_engine_and_declarations_expose_the_same_method_class():
    from increment.declarations import Method
    from increment.estimation.engine import Method as EngineMethod

    assert Method is EngineMethod
    assert Method.__module__ == "increment.declarations"


def test_method_pickle_uses_declarations_and_reads_engine_fixture():
    from increment.declarations import Method

    value = Method(name="naive")
    assert pickle.loads(pickle.dumps(value)) == value
    fixture = __import__("pathlib").Path(__file__).parent / "fixtures/method_engine_path.pickle"
    legacy = pickle.loads(fixture.read_bytes())
    assert isinstance(legacy, Method)


def test_method_invalid_variance_reduction_keeps_its_code():
    from increment.declarations import Method

    with pytest.raises(CodedError) as caught:
        Method(name="x", variance_reduction="nope")
    assert isinstance(caught.value, UnsupportedRequestError)
    assert caught.value.code == "estimation.variance.registry.no_registered_available"


@pytest.mark.integration
def test_method_classifies_builtin_observational_names_before_adjust_import():
    code = (
        "from increment.declarations import Method; "
        "assert Method(name='iptw').is_observational; "
        "assert not Method.cuped(covariates=['x']).is_observational; "
        "assert not any(n.startswith('increment.estimation') for n in __import__('sys').modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cell_key_hypothesis_conversion_for_each_cell_kind():
    from increment.declarations import (
        ArmHypothesisKey,
        CellKey,
        ContrastHypothesisKey,
        SegmentHypothesisKey,
    )

    arm = CellKey(
        kind="arm",
        metric="m",
        method="unadjusted",
        method_role="decision",
        estimand="lift",
        analysis_population="assigned",
        group_id="t",
    )
    segment = CellKey(
        kind="segment",
        metric="m",
        method="unadjusted",
        method_role="decision",
        estimand="lift",
        analysis_population="assigned",
        group_id="t",
        dimension="country",
        dimension_value="ca",
    )
    contrast = CellKey(
        kind="contrast",
        metric="m",
        method="unadjusted",
        method_role="decision",
        estimand="lift",
        analysis_population="assigned",
        control_group="c",
        treatment_group="t",
        aggregation="sum",
    )
    arm_hypothesis = arm.hypothesis()
    segment_hypothesis = segment.hypothesis()
    contrast_hypothesis = contrast.hypothesis()
    assert isinstance(arm_hypothesis, ArmHypothesisKey) and arm_hypothesis.group_id == "t"
    assert (
        isinstance(segment_hypothesis, SegmentHypothesisKey)
        and segment_hypothesis.dimension == "country"
    )
    assert isinstance(contrast_hypothesis, ContrastHypothesisKey) and (
        contrast_hypothesis.control_group,
        contrast_hypothesis.treatment_group,
    ) == ("c", "t")


def test_cell_key_serialization_roundtrip_and_mapping_immutability():
    from collections.abc import MutableMapping
    from typing import cast

    from increment.declarations import CellKey, IntegrityResult, IntegrityStatus

    key = CellKey(
        kind="arm",
        metric="m",
        method=None,
        method_role=None,
        estimand=None,
        analysis_population="assigned",
        group_id="t",
    )
    assert CellKey.model_validate_json(key.model_dump_json()) == key
    result = IntegrityResult(
        status=IntegrityStatus.FAILED,
        analysis_population="assigned",
        construction="none",
        alpha=None,
        observed={"t": 2},
        expected=None,
        randomization_grain=None,
        code=None,
        context={},
    )
    assert result.observed is not None
    observed = cast(MutableMapping[str, int], result.observed)
    with pytest.raises(TypeError):
        observed["t"] = 3


def test_cell_key_invalid_refusal_is_shared_with_readout_types():
    from increment.declarations import _CELL_KEY_INVALID_REFUSAL
    from increment.estimation.readout_types import _REFUSALS

    assert _REFUSALS["readout.cell.key_invalid"] is _CELL_KEY_INVALID_REFUSAL
    from increment.declarations import CellKey

    with pytest.raises(CodedError) as caught:
        CellKey(
            kind="arm",
            metric="m",
            method=None,
            method_role=None,
            estimand=None,
            analysis_population="assigned",
            control_group="c",
        )
    assert caught.value.code == "readout.cell.key_invalid"


def test_cell_key_and_hypothesis_read_base_pickles():
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "fixtures"
    cell = pickle.loads((fixture_dir / "cell_key_base.pickle").read_bytes())
    hypothesis = pickle.loads((fixture_dir / "hypothesis_key_base.pickle").read_bytes())
    assert cell.kind == "arm" and cell.group_id == "t"
    assert hypothesis.metric == "m" and hypothesis.group_id == "t"


@pytest.mark.integration
def test_custom_adjustment_registration_classifies_after_import():
    code = (
        "from increment.declarations import ADJUSTMENTS, Method\n"
        "ADJUSTMENTS.register('custom', lambda: [])\n"
        "assert Method(name='custom').is_observational\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_invalid_cell_key_refuses_without_loading_estimation():
    code = (
        "import sys\n"
        "from increment.declarations import CellKey\n"
        "from increment.errors import CodedError\n"
        "try:\n"
        "    CellKey(kind='arm', metric='m', method=None, method_role=None, estimand=None, "
        "analysis_population='assigned')\n"
        "except CodedError as exc:\n"
        "    assert exc.code == 'readout.cell.key_invalid'\n"
        "    assert exc.context['kind'] == 'arm'\n"
        "else:\n"
        "    raise AssertionError('invalid CellKey was accepted')\n"
        "assert not any(name.startswith('increment.estimation') for name in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_engine_importer_inventory_reports_imported_names(tmp_path):
    from scripts.engine_importers import importers

    package = tmp_path / "increment"
    package.mkdir()
    (package / "consumer.py").write_text(
        "from increment.estimation.engine import Method, estimate_lift\n"
        "from increment.estimation.engine import _df_to_arms as convert\n"
    )
    (package / "unrelated.py").write_text("from increment.errors import CodedError\n")

    assert importers(tmp_path) == {
        "increment.consumer": ("Method", "estimate_lift", "_df_to_arms"),
    }
