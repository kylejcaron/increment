"""Quantile export parity: every export-capable ingress refuses a quantile the same way.

A portable moments cube holds additive moments, never the per-unit values an order
statistic needs, so exporting a quantile would write mean moments under the quantile's
name. Every ingress refuses before a file exists, with one code (a randomized design) or
the observational-estimator code (an observational design, whose refusal takes
precedence because no estimator could use the cube). A registered sequential plan is the
one exception: it exports a finalized checkpoint of unit-record proofs, never moments rows,
so an unmodeled catalog quantile never reaches a cube.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import ibis
import pytest

from increment import Analysis
from increment.errors import CapabilityError, IncrementWarning, UnsupportedRequestError
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
from tests.warning_codes import warning_codes

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


def _mixed_observational_frame(shape: str, *, varying_covariate: bool) -> Analysis:
    """A mean precedes the quantile in the catalog; with a varying covariate it cannot be read."""
    from increment import AdjustmentSet, MetricSpec, Observational
    from tests.test_frame_panel_unit_frame import _panel_with_covariate

    quantile = MetricSpec(name="p50", type="quantile", quantile=0.5, value_column="revenue")
    if shape == "from_unit_summary":
        return Analysis.from_unit_summary(
            _confounded_table(200, seed=11),
            unit="user_id",
            group="variant",
            metrics=[MetricSpec(name="revenue", type="mean"), quantile],
            design=_OBS_TRIM,
        )
    panel, _tenure, _group = _panel_with_covariate(vary_within_unit=varying_covariate)
    return Analysis.from_unit_panel(
        panel,
        unit="user_id",
        group="variant",
        date="date",
        metrics=[MetricSpec(name="revenue", type="mean", covariate="tenure"), quantile],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
    )


@pytest.mark.parametrize(
    ("shape", "varying_covariate"),
    [("from_unit_summary", False), ("from_unit_panel", False), ("from_unit_panel", True)],
)
def test_observational_frame_refuses_a_mixed_catalog_before_reading_any_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, varying_covariate: bool
) -> None:
    """The refusal reads catalog metadata only: an earlier mean is never counted or aggregated,
    so an unreadable mean cannot pre-empt it with its own refusal."""
    from tests.analysis_factory import _moment_source

    analysis = _mixed_observational_frame(shape, varying_covariate=varying_covariate)
    source = _moment_source(analysis)
    reads: list[str] = []
    for name in ("moments", "unit_counts", "compliance_summary"):
        original = getattr(source, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            reads.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(source, name, spy)
    path = tmp_path / "mixed-moments.parquet"
    try:
        with pytest.raises(UnsupportedRequestError) as raised:
            analysis.export(path)
    finally:
        analysis.close()
    assert raised.value.code == "readout.observational.quantile"
    assert raised.value.context == {"metric": "p50"}
    assert reads == []
    assert not path.exists()


def _mixed_catalog_specs() -> list:
    from increment.frame import MetricSpec

    return [
        MetricSpec(name="outcome", type="conversion"),
        MetricSpec(name="latency", type="quantile", quantile=0.9),
    ]


def _registered_mixed_catalog(*, registered: bool) -> Analysis:
    """A modeled conversion beside a quantile no registered model covers."""
    from increment.semantics.models import AnalysisPlan
    from tests.test_sequential_public_sources import _frame, _plan

    specs = _mixed_catalog_specs()
    frame = _frame([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    frame["latency"] = [float(i % 7) for i in range(len(frame))]
    declared = pytest.warns(IncrementWarning) if registered else nullcontext()
    with declared as caught:
        plan = _plan(specs, "bernoulli") if registered else AnalysisPlan(primary="outcome")
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            control="control",
            metrics=specs,
            experiment_id="experiment",
            plan=plan,
            exposure_date="exposure",
        )
    if registered:
        assert caught is not None
        assert warning_codes(caught) == ["decision.family.quantile_always_valid_excluded"]
    return analysis


def test_registered_checkpoint_keeps_an_unmodeled_quantile_sibling_exportable(
    tmp_path: Path,
) -> None:
    """A registered checkpoint carries unit-record proofs and the declaration, no moments
    rows, so it replays exactly and never presents mean moments under the quantile's name.
    The same catalog under fixed-horizon inference refuses the moments cube."""
    import pyarrow.parquet as pq

    registered = _registered_mixed_catalog(registered=True)
    path = tmp_path / "checkpoint.parquet"
    try:
        registered.export(path)
        snapshot = registered.sequential_snapshot()
    finally:
        registered.close()
    (envelope,) = pq.read_table(path).to_pylist()
    assert envelope["record_kind"] == "sequential_checkpoint"
    assert envelope["moments_format"] == 9
    assert "metric" not in envelope
    replay = Analysis.from_moments([envelope], metrics=_mixed_catalog_specs(), control="control")
    try:
        assert replay.sequential_snapshot() == snapshot
    finally:
        replay.close()

    fixed = _registered_mixed_catalog(registered=False)
    fixed_path = tmp_path / "fixed-moments.parquet"
    try:
        with pytest.raises(CapabilityError) as raised:
            fixed.export(fixed_path)
    finally:
        fixed.close()
    assert raised.value.code == "source.frame.quantile_no_moments"
    assert raised.value.context["metric"] == "latency"
    assert not fixed_path.exists()
