"""Quantile export parity: every export-capable ingress refuses a quantile the same way.

A portable moments cube holds additive moments, never the per-unit values an order
statistic needs, so exporting a quantile would write mean moments under the quantile's
name. Every ingress refuses before a file exists, with one code (a randomized design) or
the observational-estimator code (an observational design, whose refusal takes
precedence because no estimator could use the cube).
"""

from __future__ import annotations

from pathlib import Path

import ibis
import pytest

from increment import Analysis
from increment.errors import CapabilityError, UnsupportedRequestError
from increment.query.artifact_publish import artifact_context
from increment.query.session import WarehouseArtifactStore
from increment.semantics.models import Definitions
from tests.parity_harness.cases import PARITY_CASES, _close_parity_analysis
from tests.test_native_encouragement_declaration import (
    _observational_quantile_defs,
    _seed,
    _write_defs_yaml,
)
from tests.test_readouts_observational import _OBS_TRIM, _confounded_table

_RANDOMIZED_INGRESSES = (
    "from_definitions",
    "from_unit_day_artifact",
    "from_unit_summary",
    "from_unit_panel",
)
_QUANTILE_CASES = {case.id: case for case in PARITY_CASES}


@pytest.mark.parametrize(
    "case_id", ["quantile_tied_rounded_outcomes", "audit-panel-pre-exposure-and-selection"]
)
@pytest.mark.parametrize("ingress", _RANDOMIZED_INGRESSES)
def test_randomized_quantile_export_refuses_on_every_ingress(
    tmp_path: Path, case_id: str, ingress: str
) -> None:
    """The second case also declares additive siblings: one quantile still refuses the cube."""
    analysis = _QUANTILE_CASES[case_id].build[ingress]()
    path = tmp_path / "quantile-moments.parquet"
    try:
        with pytest.raises(CapabilityError) as raised:
            analysis.export(path)
    finally:
        _close_parity_analysis(analysis)
    assert raised.value.code == "source.frame.quantile_no_moments"
    assert raised.value.context["metric"] == "p90_latency"
    assert not path.exists()


def _observational_frame(shape: str) -> Analysis:
    from increment.frame import MetricSpec

    table = _confounded_table(200, seed=11)
    metrics = [MetricSpec(name="revenue", type="quantile", quantile=0.5)]
    if shape == "from_unit_summary":
        return Analysis.from_unit_summary(
            table, unit="user_id", group="variant", metrics=metrics, design=_OBS_TRIM
        )
    import pyarrow as pa

    panel = table.append_column("day", pa.array(["2025-01-01"] * table.num_rows))
    return Analysis.from_unit_panel(
        panel, unit="user_id", group="variant", date="day", metrics=metrics, design=_OBS_TRIM
    )


@pytest.mark.parametrize("shape", ["from_unit_summary", "from_unit_panel"])
def test_observational_quantile_export_refuses_on_frame_ingresses(
    tmp_path: Path, shape: str
) -> None:
    analysis = _observational_frame(shape)
    path = tmp_path / "quantile-moments.parquet"
    try:
        with pytest.raises(UnsupportedRequestError) as raised:
            analysis.export(path)
    finally:
        analysis.close()
    assert raised.value.code == "readout.observational.quantile"
    assert raised.value.context == {"metric": "revenue"}
    assert not path.exists()


def test_observational_quantile_export_refuses_on_definitions_and_reopened_artifact(
    tmp_path: Path,
) -> None:
    """A supported mean precedes the quantile in the plan; both ingresses still refuse by name."""
    con = ibis.duckdb.connect()
    _seed(con)
    defs_dict = _observational_quantile_defs()
    defs = Definitions.model_validate(defs_dict)
    experiment = defs.experiment("native_observational_quantile_exp")
    assert experiment is not None
    defs_path = _write_defs_yaml(defs_dict, tmp_path)
    native = Analysis.from_definitions(
        "native_observational_quantile_exp", defs_path, con, store="none"
    )
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = native.publish_unit_day_artifact(store)
    reopened = Analysis.from_unit_day_artifact(
        store, ref, expected_context=artifact_context(defs, experiment, "error")
    )
    try:
        for name, analysis in (("definitions", native), ("artifact", reopened)):
            path = tmp_path / f"{name}-quantile-moments.parquet"
            with pytest.raises(UnsupportedRequestError) as raised:
                analysis.export(path)
            assert raised.value.code == "readout.observational.quantile"
            assert raised.value.context == {"metric": "revenue"}
            assert not path.exists()
    finally:
        reopened.close()
        native.close()
