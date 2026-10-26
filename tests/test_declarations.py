from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from increment.errors import CodedError


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
    assert caught.value.code == "estimation.variance.registry.no_registered_available"


@pytest.mark.integration
def test_method_classifies_builtin_observational_names_before_adjust_import():
    code = "from increment.declarations import Method; assert Method(name='iptw').is_observational; assert not any(n.startswith('increment.estimation') for n in __import__('sys').modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cell_key_hypothesis_conversion_for_each_cell_kind():
    from increment.declarations import CellKey

    common = dict(metric="m", method=None, method_role=None, estimand=None, analysis_population="assigned")
    arm = CellKey(kind="arm", group_id="t", **common)
    segment = CellKey(kind="segment", group_id="t", dimension="country", dimension_value="ca", **common)
    contrast = CellKey(kind="contrast", control_group="c", treatment_group="t", **common)
    assert arm.hypothesis().group_id == "t"
    assert segment.hypothesis().dimension == "country"
    assert (contrast.hypothesis().control_group, contrast.hypothesis().treatment_group) == ("c", "t")


def test_cell_key_serialization_roundtrip_and_mapping_immutability():
    from increment.declarations import CellKey, IntegrityResult, IntegrityStatus

    key = CellKey(kind="arm", metric="m", method=None, method_role=None, estimand=None,
                  analysis_population="assigned", group_id="t")
    assert CellKey.model_validate_json(key.model_dump_json()) == key
    result = IntegrityResult(status=IntegrityStatus.FAILED, analysis_population="assigned",
        construction="none", alpha=None, observed={"t": 2}, expected=None,
        randomization_grain=None, code=None, context={})
    with pytest.raises(TypeError):
        result.observed["t"] = 3


def test_cell_key_invalid_refusal_is_shared_with_readout_types():
    from increment.declarations import _CELL_KEY_INVALID_REFUSAL
    from increment.estimation.readout_types import _REFUSALS

    assert _REFUSALS["readout.cell.key_invalid"] is _CELL_KEY_INVALID_REFUSAL
    from increment.declarations import CellKey
    with pytest.raises(CodedError) as caught:
        CellKey(kind="arm", metric="m", method=None, method_role=None, estimand=None,
                analysis_population="assigned", control_group="c")
    assert caught.value.code == "readout.cell.key_invalid"
