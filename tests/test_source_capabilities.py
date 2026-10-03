"""Source operation refusals and runtime protocol checks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from increment._source_operations import PanelSQLOperation, SummarySqlOperation
from increment.analysis import Analysis
from increment.errors import CapabilityError
from increment.query._native_triggered import TriggeredPopulationSource
from increment.sources import (
    MomentSource,
    SourceOperation,
    require_operation,
)
from tests.test_native_core_operations import _assert_operation_refusal

if TYPE_CHECKING:
    from pydantic import BaseModel

    from increment._readout_request import ReadoutView
    from increment._source_types import Grain
    from increment.semantics.models import Metric


def test_triggered_population_source_limits_grains_and_uses_triggered_counts() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: (
            "cluster",
            {"control": 2, "treatment": 3},
            {"control": 7, "treatment": 11},
        ),
    )

    triggered = TriggeredPopulationSource(source)

    assert triggered.capabilities == frozenset({"total"})
    assert triggered.unit_counts() == {"control": 7, "treatment": 11}
    assert triggered.cluster_counts() == {"control": 2, "treatment": 3}
    with pytest.raises(CapabilityError) as raised:
        triggered.sql()
    _assert_operation_refusal(
        raised.value,
        operation="sql",
        request={"grain": "total", "population": "triggered"},
        offered=("total",),
    )


def test_triggered_population_source_uses_primary_counts_for_unclustered_units() -> None:
    source = SimpleNamespace(
        capabilities=frozenset({"total", "daily", "asof"}),
        breakouts=(),
        shape=None,
        _validate_trigger_capability=lambda *, operation: None,
        triggered_counts=lambda: ("unit", {"control": 7, "treatment": 11}, {}),
    )

    assert TriggeredPopulationSource(source).unit_counts() == {
        "control": 7,
        "treatment": 11,
    }


class _OperationsDescriptorSource:
    """Arm source whose `operations` descriptor raises when it is read."""

    context = SimpleNamespace(study_id="descriptor", design=None)
    descriptor_reads = 0

    @property
    def operations(self) -> frozenset[SourceOperation]:
        type(self).descriptor_reads += 1
        raise AttributeError("operations descriptor failed")


def test_from_source_does_not_invoke_operations_descriptor() -> None:
    source = cast(MomentSource, _OperationsDescriptorSource())

    Analysis._from_source(source)

    assert _OperationsDescriptorSource.descriptor_reads == 0
    # The descriptor is genuinely armed: touching it raises and counts.
    with pytest.raises(AttributeError):
        _ = source.operations
    assert _OperationsDescriptorSource.descriptor_reads == 1


def test_require_operation_rejects_declaration_drift() -> None:
    from tests.source_conformance import sql_totals_source

    source: MomentSource = sql_totals_source()
    assert require_operation(source, "summary_sql", SummarySqlOperation) is source
    with pytest.raises(CapabilityError) as raised:
        require_operation(source, "panel_sql", PanelSQLOperation)
    assert raised.value.code == "source.operation.unsupported"
    assert raised.value.context == {
        "operation": "panel_sql",
        "source": "SqlPanelSource",
    }


# Capability policy at the validation seam: metric type x view x option x substrate.
# Construction rules are raised by the shared gate (`GATE_REFUSALS`); source limits derive
# from each source's declared `capabilities` and `breakouts`. Axes are read from code.

_VIEW_GRAIN: dict[ReadoutView, Grain] = {
    "run": "total",
    "breakout": "total",
    "daily": "daily",
    "asof": "asof",
}
_OPTIONS = (
    "none",
    "percentile_winsorization",
    "fixed_winsorization",
    "unbounded_band",
    "cluster",
    "cuped",
)
_QUANTILE_GRAIN = "readout.metric.quantile_grain"
# Raised by `validate_readout_source` after the grain and dimension checks, so a source
# that lacks the grain reports the source limit first.
_SOURCE_STAGE_CONSTRUCTION_CODES = frozenset({_QUANTILE_GRAIN})
_DAILY_WINSOR = "readout.metric.daily_winsorization"
_PERCENTILE_WINSOR = "readout.metric.percentile_winsorization"

GATE_REFUSALS: dict[str, dict[tuple[str, str], str] | None] = {
    "mean": {
        ("daily", "percentile_winsorization"): _DAILY_WINSOR,
        ("daily", "fixed_winsorization"): _DAILY_WINSOR,
        ("asof", "percentile_winsorization"): _PERCENTILE_WINSOR,
        ("breakout", "percentile_winsorization"): _PERCENTILE_WINSOR,
    },
    "conversion": {},
    "ratio": {},
    "retention": {("daily", "unbounded_band"): "breakout.retention.unbounded"},
    "quantile": {
        ("run", "cluster"): "arm.metric.quantile_cluster",
        ("run", "cuped"): "arm.metric.quantile_cuped",
        ("breakout", "none"): "readout.metric.quantile_breakout",
        ("breakout", "cluster"): "arm.metric.quantile_cluster",
        ("breakout", "cuped"): "readout.metric.quantile_breakout",
        ("daily", "none"): _QUANTILE_GRAIN,
        ("daily", "cluster"): _QUANTILE_GRAIN,
        ("daily", "cuped"): _QUANTILE_GRAIN,
        ("asof", "none"): _QUANTILE_GRAIN,
        ("asof", "cluster"): "arm.metric.quantile_cluster",
        ("asof", "cuped"): "arm.metric.quantile_cuped",
    },
    # Report-layer types cannot be held by a source (`MATRIX["estimate"]` is NA).
    "total": None,
    "active": None,
}

# `MATRIX` capability column -> the (view, option) cell of this seam it also declares.
_MATRIX_SEAM_CELLS = {
    "breakout": (("breakout", "none"),),
    "daily_asof": (("daily", "none"), ("asof", "none")),
}


def _metric_model(metric_type: str) -> type[BaseModel]:
    from typing import get_args

    from increment.semantics import models

    union, _field = get_args(models.Metric)
    for member in get_args(union):
        if get_args(member.model_fields["type"].annotation) == (metric_type,):
            return member
    raise AssertionError(f"no semantics model for metric type {metric_type!r}")


def _option_applies(metric_type: str, option: str) -> bool:
    """Whether the metric model can express the option, read from its fields."""
    fields = _metric_model(metric_type).model_fields
    if option in ("percentile_winsorization", "fixed_winsorization"):
        return "winsorization" in fields
    if option == "unbounded_band":
        return "threshold_days" in fields
    return True


def _synthetic_metric(metric_type: str, option: str) -> Metric:
    from increment.semantics import models

    winsorization = {
        "percentile_winsorization": models.Winsorization(upper_percentile=0.99),
        "fixed_winsorization": models.Winsorization(upper_value=100.0),
    }.get(option)
    name, entity = "outcome", "user_id"
    if metric_type == "mean":
        return models.MeanMetric(
            name=name, entity=entity, fact="f", aggregation="sum", winsorization=winsorization
        )
    if metric_type == "conversion":
        return models.ConversionMetric(name=name, entity=entity, fact="f")
    if metric_type == "retention":
        return models.RetentionMetric(
            name=name,
            entity=entity,
            fact="f",
            threshold_days=1 if option == "unbounded_band" else (1, 3),
        )
    if metric_type == "ratio":
        return models.RatioMetric(
            name=name,
            entity=entity,
            numerator=models.Measure(fact="a", aggregation="sum"),
            denominator=models.Measure(fact="b", aggregation="sum"),
        )
    if metric_type == "quantile":
        return models.QuantileMetric(
            name=name, entity=entity, fact="f", aggregation="sum", quantile=0.5
        )
    raise AssertionError(
        f"metric type {metric_type!r} has no synthetic builder: declare one here and "
        "its construction refusals in GATE_REFUSALS"
    )


def _gate_request(source, metric_type: str, view: ReadoutView, option: str):
    import dataclasses

    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.estimation.engine import Method

    metric = _synthetic_metric(metric_type, option)
    context = source.context
    procedure = next(iter(context.plan.procedures.values()))
    plan = context.plan.model_copy(
        update={"procedures": {metric.name: procedure.model_copy(update={"metric": metric.name})}}
    )
    method = (
        Method(name="cuped", variance_reduction="cuped")
        if option == "cuped"
        else Method(name="unadjusted")
    )
    config = ResolvedMetricConfig(
        metric=metric,
        decision_method=method,
        sensitivity_methods=(),
        prior=None,
        prior_is_global=False,
        decision_defaulted=True,
    )
    breakouts = tuple(getattr(source, "breakouts", ()))
    dimension = (
        (breakouts[0] if breakouts else "undeclared_dimension") if view == "breakout" else None
    )
    request = ReadoutRequest.from_source(
        source,
        metrics=[metric],
        configs=[config],
        view=view,
        grain=_VIEW_GRAIN[view],
        by=(dimension,) if dimension else (),
        dimension=dimension,
    )
    return dataclasses.replace(
        request,
        context=dataclasses.replace(
            context,
            plan=plan,
            metrics=(metric,),
            configs=(config,),
            cluster="cluster_id" if option == "cluster" else context.cluster,
        ),
        cluster="cluster_id" if option == "cluster" else context.cluster,
    )


def _declared_gate_outcome(source, metric_type: str, view: ReadoutView, option: str) -> str | None:
    """The code `validate_request` must raise: construction rules, else the source's limit."""
    construction = (GATE_REFUSALS[metric_type] or {}).get((view, option))
    request_by = tuple(getattr(source, "breakouts", ()))
    if _VIEW_GRAIN[view] not in source.capabilities:
        source_limit = "readout.source.grain"
    elif view == "breakout" and not request_by:
        source_limit = "readout.source.dimension"
    else:
        source_limit = None
    if construction in _SOURCE_STAGE_CONSTRUCTION_CODES:
        return source_limit or construction
    return construction or source_limit


@pytest.fixture(scope="module")
def _substrates(tmp_path_factory: pytest.TempPathFactory):
    from tests.source_conformance import arm_adapters

    return {
        adapter.name: adapter.build()[0]
        for adapter in arm_adapters(tmp_path_factory.mktemp("substrates"))
    }


def test_gate_declarations_cover_the_code_derived_axes() -> None:
    from typing import get_args

    from increment._readout_request import ReadoutView
    from tests.compatibility_catalog import MATRIX
    from tests.test_composition_matrix import METRIC_TYPES

    assert set(GATE_REFUSALS) == set(METRIC_TYPES), (
        "decide the validation-seam policy for every metric type: "
        f"missing {set(METRIC_TYPES) - set(GATE_REFUSALS)}, "
        f"stale {set(GATE_REFUSALS) - set(METRIC_TYPES)}"
    )
    assert set(_VIEW_GRAIN) == set(get_args(ReadoutView)), (
        "give every readout view a grain and decide its policy: "
        f"{set(get_args(ReadoutView)) ^ set(_VIEW_GRAIN)}"
    )
    for metric_type, refusals in GATE_REFUSALS.items():
        report_only = MATRIX["estimate"][metric_type].status == "na"
        assert (refusals is None) == report_only, metric_type
        for view, option in refusals or {}:
            assert view in _VIEW_GRAIN and option in _OPTIONS, (metric_type, view, option)
            assert _option_applies(metric_type, option), (metric_type, view, option)


@pytest.mark.parametrize("metric_type", list(GATE_REFUSALS))
def test_gate_outcome_matches_declared_policy_on_every_substrate(
    metric_type: str, _substrates: dict[str, object]
) -> None:
    """Every registered substrate raises the declared code, or its own grain/dimension limit.

    A new metric type fails on its missing builder or `GATE_REFUSALS` entry, a new view on
    its missing grain, and a new substrate is swept with the derived rules. Options a source
    fixes at construction (sequential inference, observational and encouragement designs)
    are not request-level variants; the arm-contract sweeps in `test_refusal_uniqueness.py`
    and the pair cells in `tests/compatibility_catalog.py` own them.
    """
    from increment._readout_request import validate_request
    from increment.errors import CodedError

    if GATE_REFUSALS[metric_type] is None:
        pytest.skip("report-layer type: no source can hold it (MATRIX 'estimate' is NA)")
    mismatches = []
    for name, source in _substrates.items():
        for view in _VIEW_GRAIN:
            for option in _OPTIONS:
                if not _option_applies(metric_type, option):
                    continue
                expected = _declared_gate_outcome(source, metric_type, view, option)
                try:
                    validate_request(_gate_request(source, metric_type, view, option))
                    actual = None
                except CodedError as error:
                    actual = error.code
                if actual != expected:
                    mismatches.append((name, view, option, actual, expected))
    assert not mismatches, (
        f"{metric_type}: (substrate, view, option, actual, declared) differ: {mismatches}"
    )


def test_gate_declarations_agree_with_the_composition_matrix() -> None:
    """`MATRIX` cells for the day-axis and breakout capabilities name the same code the
    gate raises; a supported cell means the gate accepts the plain request."""
    from tests.compatibility_catalog import MATRIX

    for capability, seam_cells in _MATRIX_SEAM_CELLS.items():
        for metric_type, refusals in GATE_REFUSALS.items():
            if refusals is None:
                continue
            assert metric_type in MATRIX[capability], (
                f"MATRIX[{capability!r}] declares no cell for {metric_type!r}: decide it"
            )
            cell = MATRIX[capability][metric_type]
            for seam_cell in seam_cells:
                declared = refusals.get(seam_cell)
                if cell.status == "refused":
                    assert cell.code == declared, (capability, metric_type, seam_cell)
                else:
                    assert declared is None, (capability, metric_type, seam_cell)
