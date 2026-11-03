"""Every built-in loader answers every evidence request kind it declares alike.

One tiny event fixture is loaded once per module through each built-in loader.
Each cell of the kind x loader table either equals the hand-computed evidence
in `EXPECTED` (and the first answering loader's evidence), is an answer its
declared route gives without evidence (`EMPTY_ANSWERS`), or is a registered
refusal (`REFUSALS`). A loader never answers a kind its declaration omits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from increment import CupedCovariate, Encouragement, Method, MetricSpec, from_frame
from increment._evidence_loaders import load_contrast_stats
from increment._evidence_scopes import switchback_scope_provider
from increment._evidence_types import (
    BreakoutSlicesRequest,
    BreakoutUnitRowsRequest,
    ContrastStatsRequest,
    DayAxisRequest,
    MetricEvidenceRequest,
)
from increment.errors import CodedError
from increment.evidence import (
    AssignmentRequest,
    BreakoutIdentity,
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountsRequest,
    EvidenceCapabilities,
    EvidenceScope,
    MethodFitRequest,
    MomentsRequest,
    SequentialCaptureRequest,
    SequentialSnapshotRequest,
    SourceContext,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from increment.kernel.context import EvidenceExecutionContext
from increment.semantics.models import AnalysisPlan, Breakout, Definitions, ExperimentMetric
from tests.analysis_factory import bound_source, make_binding
from tests.evidence_fixtures.capture import _frame_rows
from tests.parity_harness import dataset
from tests.parity_harness.cases import (
    _close_parity_binding,
    _fixed_horizon_case,
    _publish_and_adopt,
    _track_connection,
)
from tests.source_conformance import encouragement_panel_analysis, switchback_analysis
from tests.test_evidence_capabilities import _registered_unit_summary_analysis

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("loader_conformance")]

LOADERS = (
    "definitions",
    "unit_day_artifact",
    "frame_summary",
    "frame_panel",
    "moments",
    "switchback_frame",
)
_ARMS = ("control", "treatment")
_STORE = BreakoutIdentity("store", None)
_UNADJUSTED = Method(name="unadjusted")
_STORE_CUPED = Method.cuped_multicovariate(
    name="mixed", covariates=CupedCovariate(name="store", kind="categorical")
)
_ENCOURAGEMENT = Encouragement.model_validate(
    {
        "mechanism": "encouragement",
        "control_group": "control",
        "uptake": {"fact": "converted"},
        "exclusion_restriction": {"justification": "the prompt has no direct effect"},
    }
)
# Evidence-level tolerances: moments 1e-9, other floats 1e-12, both relative
# with an absolute floor of the same size.
_MOMENTS_TOLERANCE = 1e-9
_VALUE_TOLERANCE = 1e-12


def _declared_kinds(evidence: EvidenceCapabilities) -> frozenset[type]:
    """Request kinds one declaration says its loader can answer."""
    kinds: set[type] = set()
    if "total" in evidence.grains:
        kinds |= {MetricEvidenceRequest, MomentsRequest, EvidenceScope}
    if evidence.grains & {"daily", "asof"}:
        kinds.add(DayAxisRequest)
    if evidence.unit_rows:
        kinds.add(UnitRowsRequest)
    if evidence.count_grains:
        kinds.add(CountsRequest)
    if evidence.assignment_relation != "none":
        kinds.add(AssignmentRequest)
    if evidence.method_fit != "none":
        kinds.add(MethodFitRequest)
    if evidence.compliance_axis != "none":
        kinds.add(ComplianceRequest)
    if evidence.compliance_axis == "asof":
        kinds.add(ComplianceSeriesRequest)
    if evidence.sequential_capture != "none":
        kinds |= {SequentialCaptureRequest, SequentialSnapshotRequest}
    if evidence.breakouts:
        kinds.add(BreakoutSlicesRequest)
        if evidence.unit_rows:
            kinds.add(BreakoutUnitRowsRequest)
        if evidence.scalar is not None and evidence.scalar.breakout_evidence:
            kinds.add(BreakoutMethodMomentsRequest)
    if evidence.kind == "switchback":
        kinds.add(ContrastStatsRequest)
    return frozenset(kinds)


# -- Hand-computed evidence -------------------------------------------------

# Six units per arm exposed 2025-01-10 09:00 with a 24-hour window: i = 1, 2, 4, 5
# buy 5 + i % 5 inside it (one more in treatment), i = 0, 3 never buy, and every
# unit ends one session; all other rows fall outside. Store is s0 for i < 5, s1
# for i = 5.

_REVENUE = {
    "control": (0.0, 6.0, 7.0, 0.0, 9.0, 5.0),
    "treatment": (0.0, 7.0, 8.0, 0.0, 10.0, 6.0),
}
_CONVERTED = {
    "control": (0.0, 1.0, 1.0, 0.0, 1.0, 1.0),
    "treatment": (0.0, 1.0, 1.0, 0.0, 1.0, 1.0),
}
_STORES = ("s0", "s0", "s0", "s0", "s0", "s1")
_ONE_SESSION = {"mean": 1.0, "m2": 0.0, "m3": 0.0, "c_y": 0.0}


def _m(n, total, squares, *, successes=None, den=None, x=None):
    """Moments of n values with the given sum and sum of squares."""
    return {
        "n": n,
        "successes": successes,
        "mean": total / n,
        "m2": squares - total**2 / n,
        "den": den,
        "x": x,
        "extra": (),
    }


def _day_rows(days):
    return {
        metric: {(day, arm): row for day in days for arm, row in rows.items()}
        for metric, rows in _TOTALS.items()
    }


# The as-of view reports each calendar day from the experiment start through its
# declared end; every window closes 2025-01-11 09:00, so each day carries the totals.
_WAREHOUSE_ASOF_DAYS = tuple(f"2025-01-{day:02d}" for day in range(10, 21))
# A frame panel's as-of axis is its own day labels (docs/guides/encouragement.md:277,
# docs/guides/sequential-inference.md:460): exposure, right-edge and padding days.
_PANEL_ASOF_DAYS = ("2025-01-10", "2025-01-11", "2025-01-19")
_DAILY_SLICE = {"view": "daily", "dimension": None, "source_name": None, "observation_edges": None}
_ASOF_SLICE = {"view": "asof", "dimension": None, "source_name": None, "observation_edges": None}


# Revenue: control 0 + 6 + 7 + 0 + 9 + 5 = 27 (squares 191), treatment
# 0 + 7 + 8 + 0 + 10 + 6 = 31 (squares 249). Four of six units convert.
_TOTALS = {
    "revenue": {"control": _m(6, 27, 191), "treatment": _m(6, 31, 249)},
    "purchase_rate": {arm: _m(6, 4, 4, successes=4) for arm in _ARMS},
    "rps": {
        "control": _m(6, 27, 191, den=_ONE_SESSION),
        "treatment": _m(6, 31, 249, den=_ONE_SESSION),
    },
}
# s0 holds units 0..4 (control 0, 6, 7, 0, 9; treatment 0, 7, 8, 0, 10); s1
# holds unit 5 alone (control 5, treatment 6).
_BY_STORE = {
    "revenue": {
        ("s0", "control"): _m(5, 22, 166),
        ("s0", "treatment"): _m(5, 25, 213),
        ("s1", "control"): _m(1, 5, 25),
        ("s1", "treatment"): _m(1, 6, 36),
    },
    "purchase_rate": {
        **{("s0", arm): _m(5, 3, 3, successes=3) for arm in _ARMS},
        **{("s1", arm): _m(1, 1, 1, successes=1) for arm in _ARMS},
    },
    "rps": {
        ("s0", "control"): _m(5, 22, 166, den=_ONE_SESSION),
        ("s0", "treatment"): _m(5, 25, 213, den=_ONE_SESSION),
        ("s1", "control"): _m(1, 5, 25, den=_ONE_SESSION),
        ("s1", "treatment"): _m(1, 6, 36, den=_ONE_SESSION),
    },
}


def _units(values, *, den=False, store=False):
    return {
        f"{arm[0]}{index}": {
            "group_id": arm,
            "y": value,
            **({"y_den": 1.0} if den else {}),
            **({"store": _STORES[index]} if store else {}),
        }
        for arm, arm_values in values.items()
        for index, value in enumerate(arm_values)
    }


EXPECTED: dict[str, object] = {
    "MetricEvidenceRequest": {
        metric: {
            "rows": rows,
            "unit_frame": None,
            "observed_arms": _ARMS,
            "n_treatment_arms": 1,
            "control_group": "control",
            "cluster": None,
            "evidence_kind": "moments_rows",
            "method_rows": {},
            "method_refusals": (),
            "method_cell_refusals": (),
            "method_fit_answers": (),
        }
        for metric, rows in _TOTALS.items()
    },
    "MomentsRequest": _TOTALS,
    # The scope pins a view that answers the same totals.
    "EvidenceScope": _TOTALS,
    # A one-day window gives every unit one window day: its exposure day.
    "DayAxisRequest": {
        "daily": {"slices": (_DAILY_SLICE,), "rows": _day_rows(("2025-01-10",))},
        "asof": {"slices": (_ASOF_SLICE,), "rows": _day_rows(_WAREHOUSE_ASOF_DAYS)},
    },
    "UnitRowsRequest": {
        "revenue": _units(_REVENUE),
        "purchase_rate": _units(_CONVERTED),
        "rps": _units(_REVENUE, den=True),
    },
    "CountsRequest": {
        "population": "assigned",
        "grain": "unit",
        "counts": {"control": 6, "treatment": 6},
        "unit_counts": {},
    },
    "AssignmentRequest": {
        "same_experiment": True,
        "population": "assigned",
        "versions": (1,),
        "periods": (None,),
        "arm_counts": ((1, None, "control", 6), (1, None, "treatment", 6)),
        "independence_grain": "unit",
        "probabilities": (),
        "replay_status": "not_certified",
        "order_draws": (),
        "randomization_law": None,
    },
    "BreakoutSlicesRequest": {"store": {"source_name": None, "moments": _BY_STORE}},
    "BreakoutUnitRowsRequest": {
        metric: {"breakout": "store", "has_dimension": True, "units": units}
        for metric, units in (
            ("revenue", _units(_REVENUE, store=True)),
            ("purchase_rate", _units(_CONVERTED, store=True)),
            ("rps", _units(_REVENUE, den=True, store=True)),
        )
    },
    "BreakoutMethodMomentsRequest": {
        metric: {"rows": rows, "refusals": ()} for metric, rows in _BY_STORE.items()
    },
    # u1 runs control then treatment, u2 the reverse; each period's one observed step
    # reads cycle + period + 1, plus 1 in treatment. u1's cycles differ by 2 and u2's
    # by 0: unit means 2 and 0 (total 2, M2 2) around the first unit's 2; each cycle
    # adds 1 to every value.
    "ContrastStatsRequest": {
        "metric": "value",
        "aggregation": "sum",
        "probability_ct": 0.5,
        "randomization_law": "independent_bernoulli_order",
        "independence_grain": "unit_cycle",
        "washout_steps": 1,
        "carryover_order": 0,
        "observation_steps": 1,
        "retained_steps": 1,
        "identifying_assumption": "no_residual_carryover_after_discarded_steps",
        "control_group": "control",
        "treatment_group": "treatment",
        "n_units": 2,
        "n_cycles": 4,
        "n_blocks": None,
        "reference_delta": 2.0,
        "mean_residual": -1.0,
        "m2_delta": 2.0,
        "mean_slope": 1.0,
        "ct_cycles": 2,
        "tc_cycles": 2,
        "minimum_cycles_per_unit": 2,
        "maximum_cycles_per_unit": 2,
        "exact_delta_total": (2, 1),
    },
}

# Declared routes that answer differently: a unit summary serves breakouts as method
# moments and unit rows, not slice views; a local-fit loader leaves a multicovariate
# fit to its consumer; a frame panel's as-of axis is its own day labels.
ROUTE_ANSWERS: dict[tuple[str, str], object] = {
    ("BreakoutSlicesRequest", "frame_summary"): {},
    ("MethodFitRequest", "frame_summary"): {
        "origin": "unavailable",
        "envelope": None,
        "cells": None,
    },
    ("MethodFitRequest", "frame_panel"): {"origin": "unavailable", "envelope": None, "cells": None},
    ("DayAxisRequest", "frame_panel"): {
        "daily": {"slices": (_DAILY_SLICE,), "rows": _day_rows(("2025-01-10",))},
        "asof": {"slices": (_ASOF_SLICE,), "rows": _day_rows(_PANEL_ASOF_DAYS)},
    },
}


# -- Registered refusals -----------------------------------------------------


def _grain(constructor, requested, offered):
    return (
        "evidence.grain_unavailable",
        {
            "constructor": constructor,
            "requested": requested,
            "offered": offered,
            "route": "use a source that retains the requested evidence",
        },
    )


def _operation(constructor, requested, offered, route):
    return (
        "evidence.operation_unavailable",
        {"constructor": constructor, "requested": requested, "offered": offered, "route": route},
    )


def _no_unit_rows(constructor):
    return (
        "evidence.unit_rows_unavailable",
        {
            "constructor": constructor,
            "requested": True,
            "offered": False,
            "route": "use a source that retains unit-level rows",
        },
    )


def _no_breakout(constructor):
    return (
        "evidence.breakout_unavailable",
        {
            "constructor": constructor,
            "requested": ("store",),
            "offered": (),
            "route": "use a source with the requested breakout dimensions",
        },
    )


_CONSTRUCTORS = {
    "definitions": "DefinitionsMomentSource",
    "unit_day_artifact": "_ArtifactFacadeSource",
    "frame_summary": "FrameTotalsSource",
    "frame_panel": "FramePanelSource",
    "moments": "MomentsSource",
    "switchback_frame": "FrameSwitchbackSource",
}
_SWITCHBACK = _CONSTRUCTORS["switchback_frame"]
_COMPLIANCE_ROUTE = "use a source with the requested compliance axis and population"
_PANEL_UNIT_FRAME = (
    "source.frame.unit_frame_panel",
    {
        "metric": "revenue",
        "shape": "a unit panel",
        "route": (
            "compute each unit's windowed value upstream and declare it as an "
            "unwindowed metric on from_unit_summary"
        ),
    },
)
_ARTIFACT_STORE_COVARIATE = (
    "artifact.extension.missing",
    {
        "covariate": "store",
        "metric": "revenue",
        "route": "republish the artifact from the definitions that declare this CUPED covariate",
    },
)
_NO_UPTAKE = (
    "source.frame.compliance_no_uptake",
    {
        "source": "FramePanelSource",
        "route": "construct from_unit_panel with an Encouragement design or uptake=",
    },
)
_NO_CHECKPOINT = (
    "sequential.continuation.legacy",
    {
        "reason": (
            "source has no exact finalized joint-unit checkpoint; capture or adopt a "
            "current sequential snapshot"
        )
    },
)
_UNREGISTERED = (
    "sequential.source.invalid",
    {"reason": "register the likelihood before capturing source data"},
)
_UNROUTED = (
    "evidence.request_unsupported",
    {
        "constructor": "_BuiltInEvidenceProvider",
        "requested": "ContrastStatsRequest",
        "offered": (
            "BreakoutMethodMomentsRequest",
            "ComplianceRequest",
            "ComplianceSeriesRequest",
            "CountsRequest",
            "EvidenceScope",
            "MethodFitRequest",
            "MomentsRequest",
            "SequentialCaptureRequest",
            "UnitRowsRequest",
        ),
        "route": "use one of the supported evidence request types",
    },
)

REFUSALS: dict[tuple[str, str], tuple[str, Mapping[str, object]]] = {
    ("MetricEvidenceRequest", "switchback_frame"): _grain(_SWITCHBACK, "total", ()),
    ("MomentsRequest", "switchback_frame"): _grain(_SWITCHBACK, "total", ()),
    ("EvidenceScope", "switchback_frame"): _grain(_SWITCHBACK, "total", ()),
    ("DayAxisRequest", "frame_summary"): _grain("FrameTotalsSource", "daily", ("total",)),
    ("DayAxisRequest", "moments"): _grain("MomentsSource", "daily", ("total",)),
    ("DayAxisRequest", "switchback_frame"): _grain(_SWITCHBACK, "daily", ()),
    ("UnitRowsRequest", "frame_panel"): _PANEL_UNIT_FRAME,
    ("UnitRowsRequest", "moments"): _no_unit_rows("MomentsSource"),
    ("UnitRowsRequest", "switchback_frame"): _no_unit_rows(_SWITCHBACK),
    ("CountsRequest", "switchback_frame"): _operation(
        _SWITCHBACK, "counts", (), "use a source that retains the requested counts"
    ),
    ("AssignmentRequest", "switchback_frame"): (
        "evidence.assignment.selection_unavailable",
        {
            "constructor": _SWITCHBACK,
            "requested": {"period": None},
            "offered": {
                "periods": ("0:period=0", "0:period=1", "1:period=0", "1:period=1"),
                "route": "set all_periods=True",
            },
        },
    ),
    # Store level s1 holds one unit per arm, too few for a categorical fit.
    ("MethodFitRequest", "definitions"): (
        "estimation.cuped_linear.level_support",
        {"column": "store", "level": "s1", "counts": {"control": 1, "treatment": 1}},
    ),
    ("MethodFitRequest", "unit_day_artifact"): _ARTIFACT_STORE_COVARIATE,
    **{
        ("MethodFitRequest", loader): _operation(
            _CONSTRUCTORS[loader],
            "method_fit",
            ("warehouse", "local", "retained"),
            "use a source that retains method-specific evidence",
        )
        for loader in ("moments", "switchback_frame")
    },
    **{
        ("ComplianceRequest", loader): _operation(
            _CONSTRUCTORS[loader], "compliance", ("total", "asof"), _COMPLIANCE_ROUTE
        )
        for loader in LOADERS
        if loader != "frame_panel"
    },
    ("ComplianceRequest", "frame_panel"): _NO_UPTAKE,
    **{
        ("ComplianceSeriesRequest", loader): _operation(
            _CONSTRUCTORS[loader], "compliance_series", ("asof",), _COMPLIANCE_ROUTE
        )
        for loader in LOADERS
        if loader != "frame_panel"
    },
    ("ComplianceSeriesRequest", "frame_panel"): _NO_UPTAKE,
    ("SequentialCaptureRequest", "definitions"): _UNREGISTERED,
    ("SequentialCaptureRequest", "unit_day_artifact"): _UNREGISTERED,
    ("SequentialCaptureRequest", "frame_panel"): (
        "sequential.source.invalid",
        {"reason": "panel capture needs registration, explicit finalization and exposure anchors"},
    ),
    **{
        ("SequentialCaptureRequest", loader): _operation(
            _CONSTRUCTORS[loader],
            "capture_sequential",
            ("construction", "asof", "replay"),
            "use a source with sequential history",
        )
        for loader in ("frame_summary", "moments", "switchback_frame")
    },
    **{
        ("SequentialSnapshotRequest", loader): _NO_CHECKPOINT
        for loader in LOADERS
        if loader != "switchback_frame"
    },
    ("SequentialSnapshotRequest", "switchback_frame"): (
        "sequential.continuation.legacy",
        {
            "reason": (
                "this source has no exact finalized checkpoint; use a raw-record, frame, "
                "native or current artifact capture"
            )
        },
    ),
    ("BreakoutSlicesRequest", "moments"): _no_breakout("MomentsSource"),
    ("BreakoutSlicesRequest", "switchback_frame"): _no_breakout(_SWITCHBACK),
    ("BreakoutUnitRowsRequest", "unit_day_artifact"): _ARTIFACT_STORE_COVARIATE,
    ("BreakoutUnitRowsRequest", "frame_panel"): _PANEL_UNIT_FRAME,
    ("BreakoutUnitRowsRequest", "moments"): _no_unit_rows("MomentsSource"),
    ("BreakoutUnitRowsRequest", "switchback_frame"): _no_unit_rows(_SWITCHBACK),
    **{
        ("BreakoutMethodMomentsRequest", loader): _operation(
            _CONSTRUCTORS[loader],
            "breakout_method_moments",
            ("method-specific breakout moments",),
            "use a source that retains method-specific breakout evidence",
        )
        for loader in ("moments", "switchback_frame")
    },
    # Parallel-arm loaders hold no switchback contrast; the typed boundary refuses it.
    **{
        ("ContrastStatsRequest", loader): _UNROUTED
        for loader in LOADERS
        if loader != "switchback_frame"
    },
}


# -- Acquisition and projection ----------------------------------------------

# Fixture vocabulary that must stay null for an unwinsorized, undiluted metric.
_EXTRA = (
    "sum_d",
    "cyd",
    "cy2d",
    "cxd",
    "winsor_lower_percentile",
    "winsor_upper_percentile",
    "winsor_lower_bound",
    "winsor_upper_bound",
    "winsor_n",
    "winsor_n_lower",
    "winsor_n_upper",
)


def _plain(value: Any) -> Any:
    if isinstance(value, CodedError):
        return (value.code, _context(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, BaseModel):
        return {name: _plain(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_plain(item) for item in value))
    return value


def _context(refusal: CodedError) -> dict[str, object]:
    # A rendered message carried in context is wording, not contract.
    return {key: _plain(value) for key, value in refusal.context.items() if key != "message"}


@dataclass(frozen=True, slots=True)
class Refused:
    code: str
    context: Mapping[str, object]


def _ask(source: Any, request: object) -> Any:
    return request_evidence(evidence_provider(source), request)


def _moments(row: Mapping[str, Any]) -> dict[str, object]:
    """Reference-free statistics of one centered row."""
    n = row["n"]
    den = None
    if row.get("ref_den") is not None:
        shift = row["cden1"] / n
        den = {
            "mean": row["ref_den"] + shift,
            "m2": row["cden2"] - row["cden1"] * shift,
            "m3": None
            if row.get("cden3") is None
            else row["cden3"] - 3 * shift * row["cden2"] + 2 * n * shift**3,
            "c_y": row["cyden"] - row["cy1"] * shift,
        }
    x = None
    if any(row.get(name) is not None for name in ("x_role", "ref_x", "cx1", "cx2", "cxy")):
        x_shift = row["cx1"] / n
        x = {
            "role": row.get("x_role"),
            "mean": row["ref_x"] + x_shift,
            "m2": row["cx2"] - row["cx1"] * x_shift,
            "c_y": row["cxy"] - row["cy1"] * x_shift,
            "c_den": None if row.get("cxden") is None else row["cxden"] - row["cden1"] * x_shift,
        }
    return {
        "n": n,
        "successes": row.get("successes"),
        "mean": row["ref_y"] + row["cy1"] / n,
        "m2": row["cy2"] - row["cy1"] ** 2 / n,
        "den": den,
        "x": x,
        "extra": tuple(name for name in _EXTRA if row.get(name) is not None),
    }


def _moment_rows(rows, by: tuple[str, ...] = ()) -> dict[object, object]:
    projected: dict[object, object] = {}
    for row in rows:
        day = row.get("ds")
        parts = (
            *(() if day is None else (str(day),)),
            *(row.get(column) for column in by),
            str(row["group_id"]),
        )
        key = parts[0] if len(parts) == 1 else parts
        assert key not in projected, f"duplicate evidence row {key}"
        projected[key] = _moments(row)
    return projected


def _unit_rows_of(frame) -> dict[str, object]:
    units: dict[str, object] = {}
    for row in _frame_rows(frame):
        unit = str(row.pop("unit_id"))
        assert unit not in units, f"duplicate unit row {unit}"
        units[unit] = row
    return units


def _metric_evidence(source: Any) -> object:
    answers = {}
    for metric in source.context.metrics:
        evidence = _ask(
            source,
            MetricEvidenceRequest(metric=metric, control_group="control", methods=(_UNADJUSTED,)),
        )
        answers[metric.name] = {
            "rows": _moment_rows(evidence.rows),
            "unit_frame": None
            if evidence.unit_frame is None
            else _unit_rows_of(evidence.unit_frame),
            "observed_arms": tuple(sorted(evidence.observed_arms)),
            "n_treatment_arms": evidence.n_treatment_arms,
            "control_group": evidence.control_group,
            "cluster": evidence.cluster,
            "evidence_kind": evidence.evidence_kind,
            "method_rows": {name: _moment_rows(rows) for name, rows in evidence.method_rows},
            "method_refusals": tuple(
                (name, spec.code, _plain(context))
                for name, spec, context in evidence.method_refusals
            ),
            "method_cell_refusals": tuple(
                (name, dimension, value, spec.code, _plain(context))
                for name, dimension, value, spec, context in evidence.method_cell_refusals
            ),
            "method_fit_answers": tuple(
                (request.method.name, answer.origin)
                for request, answer in evidence.method_fit_answers
            ),
        }
    return answers


def _totals(source: Any) -> object:
    return {
        metric.name: _moment_rows(_ask(source, MomentsRequest(metric)))
        for metric in source.context.metrics
    }


def _scope(source: Any) -> object:
    metrics = tuple(source.context.metrics)
    scoped = _ask(source, EvidenceScope(metrics=metrics, grain="total"))
    with scoped.source as pinned:
        return _totals(pinned)


def _day_axis(source: Any) -> object:
    metrics = tuple(source.context.metrics)
    return {
        grain: _day_axis_evidence(
            _ask(
                source, DayAxisRequest(caller=caller, grain=grain, metrics=metrics, dimension=None)
            )
        )
        for grain, caller in (("daily", "run_daily"), ("asof", "run_asof"))
    }


def _day_axis_evidence(answer: Any) -> object:
    rows: dict[str, list[Mapping[str, Any]]] = {}
    slices: dict[str, object] = {}
    for item in answer.slices:
        described = {
            "view": item.view,
            "dimension": item.dimension,
            "source_name": item.source_name,
            "observation_edges": _plain(item.observation_edges),
        }
        slices[repr(described)] = described
        for row in item.rows:
            rows.setdefault(str(row["metric"]), []).append(row)
    return {
        "slices": tuple(slices[key] for key in sorted(slices)),
        "rows": {metric: _moment_rows(metric_rows) for metric, metric_rows in rows.items()},
    }


def _unit_rows(source: Any) -> object:
    return {
        metric.name: _unit_rows_of(_ask(source, UnitRowsRequest(metric)))
        for metric in source.context.metrics
    }


def _counts(source: Any) -> object:
    answer = _ask(source, CountsRequest(kind="unit"))
    return {
        "population": answer.population,
        "grain": answer.grain,
        "counts": {str(arm): count for arm, count in answer.counts.items()},
        "unit_counts": {str(arm): count for arm, count in answer.unit_counts.items()},
    }


def _assignment(source: Any) -> object:
    request = AssignmentRequest(experiment=source.context.study_id)
    answer = _ask(source, request)
    return {
        "same_experiment": answer.experiment == request.experiment,
        **{
            item.name: _plain(getattr(answer, item.name))
            for item in fields(answer)
            if item.name != "experiment"
        },
    }


def _store_fit(source: Any) -> Any:
    """Request the store-adjusted fit of the source's first metric; raise a coded failure."""
    answer = _ask(
        source,
        MethodFitRequest(
            metric=source.context.metrics[0],
            method=_STORE_CUPED,
            control="control",
            arm_order=_ARMS,
            population="assigned",
        ),
    )
    if answer.failure is not None:
        raise answer.failure
    return answer


def _method_fit(source: Any) -> object:
    answer = _store_fit(source)
    return {
        "origin": answer.origin,
        "envelope": _plain(answer.envelope),
        "cells": _plain(answer.cells),
    }


def _breakout_slices(source: Any) -> object:
    metrics = tuple(source.context.metrics)
    # Slice views resolve each metric's declared configuration from the request.
    configs = tuple(source.context.configs) if isinstance(source.context, SourceContext) else ()
    request = BreakoutSlicesRequest(
        breakouts=(_STORE,),
        breakout_definitions=(Breakout(property="store"),),
        metrics=metrics,
        configs=configs,
    )
    return _slice_evidence(_ask(source, request), metrics)


def _slice_evidence(answer: Any, metrics: tuple[Any, ...]) -> object:
    slices: dict[str, object] = {}
    for item in answer.slices:
        assert item.identity.property not in slices, f"duplicate slice {item.identity}"
        slices[item.identity.property] = {
            "source_name": item.source_name,
            "moments": {
                metric.name: _moment_rows(
                    _ask(item.source, MomentsRequest(metric, by=("store",))), by=("store",)
                )
                for metric in metrics
            },
        }
    return slices


def _contrast_stats(source: Any) -> object:
    metric = source.context.metrics[0]
    request = ContrastStatsRequest(metric=metric)
    if source.evidence.kind != "switchback":
        return _ask(source, request)
    # A switchback contrast is acquired through a run's borrowed evidence scope.
    context = EvidenceExecutionContext(
        provider=switchback_scope_provider(source),
        scope=EvidenceScope(grain="total", population="assigned", metrics=(metric,)),
        run_id="loader-conformance",
    )
    with context.evidence_scope():
        return _plain(load_contrast_stats(source, request, execution_context=context))


def _breakout_unit_rows(source: Any) -> object:
    answers = {}
    for metric in source.context.metrics:
        answer = _ask(source, BreakoutUnitRowsRequest(metric=metric, breakout=_STORE))
        answers[metric.name] = {
            "breakout": answer.identity.property,
            "has_dimension": answer.unit_frame_has_dimension,
            "units": _unit_rows_of(answer.frame),
        }
    return answers


def _breakout_method_moments(source: Any) -> object:
    answers = {}
    for metric in source.context.metrics:
        answer = _ask(
            source,
            BreakoutMethodMomentsRequest(metric=metric, method=_UNADJUSTED, breakout=_STORE),
        )
        answers[metric.name] = {
            "rows": _moment_rows(answer.rows, by=("store",)),
            "refusals": _plain(answer.refusals),
        }
    return answers


@dataclass(frozen=True, slots=True)
class _Kind:
    acquire: Callable[[Any], object]
    tolerance: float


_KINDS: dict[type, _Kind] = {
    MetricEvidenceRequest: _Kind(_metric_evidence, _MOMENTS_TOLERANCE),
    MomentsRequest: _Kind(_totals, _MOMENTS_TOLERANCE),
    EvidenceScope: _Kind(_scope, _MOMENTS_TOLERANCE),
    DayAxisRequest: _Kind(_day_axis, _MOMENTS_TOLERANCE),
    UnitRowsRequest: _Kind(_unit_rows, _VALUE_TOLERANCE),
    CountsRequest: _Kind(_counts, _VALUE_TOLERANCE),
    AssignmentRequest: _Kind(_assignment, _VALUE_TOLERANCE),
    MethodFitRequest: _Kind(_method_fit, _VALUE_TOLERANCE),
    ComplianceRequest: _Kind(
        lambda source: _plain(_ask(source, ComplianceRequest(design=_ENCOURAGEMENT))),
        _VALUE_TOLERANCE,
    ),
    ComplianceSeriesRequest: _Kind(
        lambda source: _plain(_ask(source, ComplianceSeriesRequest(design=_ENCOURAGEMENT))),
        _VALUE_TOLERANCE,
    ),
    SequentialCaptureRequest: _Kind(
        lambda source: _ask(
            source, SequentialCaptureRequest(finalized=True, as_of=date(2025, 1, 20))
        ),
        _VALUE_TOLERANCE,
    ),
    SequentialSnapshotRequest: _Kind(
        lambda source: _ask(source, SequentialSnapshotRequest()), _VALUE_TOLERANCE
    ),
    BreakoutSlicesRequest: _Kind(_breakout_slices, _MOMENTS_TOLERANCE),
    BreakoutUnitRowsRequest: _Kind(_breakout_unit_rows, _VALUE_TOLERANCE),
    BreakoutMethodMomentsRequest: _Kind(_breakout_method_moments, _MOMENTS_TOLERANCE),
    ContrastStatsRequest: _Kind(_contrast_stats, _VALUE_TOLERANCE),
}
CONFORMANCE_KINDS: tuple[type, ...] = tuple(_KINDS)


def _assert_close(actual: Any, expected: Any, tolerance: float, path: str) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        for value in (actual, expected):
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (path, value)
        assert abs(actual - expected) <= tolerance * max(1.0, abs(expected)), (
            path,
            actual,
            expected,
        )
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), (path, actual)
        assert actual.keys() == expected.keys(), (
            path,
            sorted(map(repr, actual.keys() - expected.keys())),
            sorted(map(repr, expected.keys() - actual.keys())),
        )
        for key in expected:
            _assert_close(actual[key], expected[key], tolerance, f"{path}[{key!r}]")
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected), (path, actual, expected)
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(left, right, tolerance, f"{path}[{index}]")
    else:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)


# -- Fixture -------------------------------------------------------------------


def _event_rows() -> list[dict[str, Any]]:
    # A nonzero purchase and a session exactly at each unit's excluded right
    # window edge change the evidence if a one-day window is read as two.
    rows = dataset.event_rows(n_per_arm=6)
    return rows + [
        {
            **row,
            "event": event,
            "event_at": datetime(2025, 1, 11, 9),
            "revenue": 100.0 if event == "purchase" else None,
            "sess": 1 if event == "session_end" else None,
        }
        for row in rows
        if row["event"] == "exposure"
        for event in ("purchase", "session_end")
    ]


def _frame_panel(rows: list[dict[str, Any]]) -> tuple[Any, Any]:
    con = dataset.duckdb_connection(rows)
    try:
        panel = dataset.sequential_unit_panel_frame(con)
    finally:
        con.disconnect()
    panel["converted"] = (panel["revenue"] > 0).astype(int)
    panel["store"] = [dataset.store_for(int(unit[1:])) for unit in panel["user_id"]]
    return from_frame(
        panel,
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        exposure_date="exposure_date",
        observation_end="2025-01-20",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue", missing="zero", window_days=1, preferred_direction="increase"
            ),
            MetricSpec(
                name="purchase_rate",
                type="conversion",
                value_column="converted",
                window_days=1,
                preferred_direction="increase",
            ),
            MetricSpec(
                name="rps",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
                missing="zero",
                window_days=1,
                preferred_direction="increase",
            ),
        ],
        plan=AnalysisPlan(q=0.10, secondaries=("revenue", "purchase_rate", "rps")),
        breakouts=("store",),
    )


@dataclass(frozen=True)
class _Loaded:
    sources: Mapping[str, Any]
    # Each cell is acquired once; later cells compare against the cached answer.
    _outcomes: dict[tuple[type, str], object] = field(default_factory=dict)

    def outcome(self, kind: type, loader: str) -> object:
        key = (kind, loader)
        if key not in self._outcomes:
            try:
                self._outcomes[key] = _KINDS[kind].acquire(self.sources[loader])
            except CodedError as refusal:
                self._outcomes[key] = Refused(refusal.code, _context(refusal))
        return self._outcomes[key]


@pytest.fixture(scope="module")
def loaded() -> Iterator[_Loaded]:
    """Each built-in loader over the same tiny fixture; the schedule-shaped
    switchback loader has its own, since parallel arms have no cycles."""
    rows = _event_rows()
    bound: dict[str, tuple[Any, Any]] = {}
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dataset, "event_rows", lambda **_: [dict(row) for row in rows])
            case = _fixed_horizon_case(id="loader-conformance", breakout=True, include_cuped=False)
            builds = {
                "definitions": case.build["from_definitions"],
                "unit_day_artifact": case.build["from_unit_day_artifact"],
                "frame_summary": case.build["from_unit_summary"],
                "frame_panel": lambda: _frame_panel(rows),
                "moments": case.build["from_moments"],
                "switchback_frame": switchback_analysis,
            }
            for name in LOADERS:
                bound[name] = builds[name]()
        yield _Loaded(
            MappingProxyType(
                {name: bound_source(resolver) for name, (_, resolver) in bound.items()}
            )
        )
    finally:
        for pair in bound.values():
            _close_parity_binding(pair)


def declared_request_kinds_of_builtin_loaders(sources: Mapping[str, Any]) -> set[type]:
    """The union of request kinds the built-in loaders' declarations answer."""
    return set().union(*(_declared_kinds(source.evidence) for source in sources.values()))


def test_conformance_covers_every_declared_request_kind(loaded):
    assert set(CONFORMANCE_KINDS) == declared_request_kinds_of_builtin_loaders(loaded.sources)
    cells = {(kind.__name__, loader) for kind in CONFORMANCE_KINDS for loader in LOADERS}
    assert REFUSALS.keys() <= cells and ROUTE_ANSWERS.keys() <= cells
    assert not REFUSALS.keys() & ROUTE_ANSWERS.keys()
    assert EXPECTED.keys() <= {kind.__name__ for kind in CONFORMANCE_KINDS}


@pytest.mark.grid("one declared contract per evidence request kind and built-in loader")
@pytest.mark.parametrize("loader", LOADERS)
@pytest.mark.parametrize("kind", CONFORMANCE_KINDS, ids=lambda kind: kind.__name__)
def test_loader_answers_request_kind(kind, loader, loaded):
    cell = (kind.__name__, loader)
    outcome = _without_known_difference(cell, loaded.outcome(kind, loader))
    if cell in REFUSALS:
        code, context = REFUSALS[cell]
        assert outcome == Refused(code, context)
        return
    assert kind in _declared_kinds(loaded.sources[loader].evidence), (cell, outcome)
    assert not isinstance(outcome, Refused), (cell, outcome)
    tolerance = _KINDS[kind].tolerance
    if cell in ROUTE_ANSWERS:
        _assert_close(outcome, ROUTE_ANSWERS[cell], tolerance, str(cell))
        return
    assert kind.__name__ in EXPECTED, (cell, outcome)
    _assert_close(outcome, EXPECTED[kind.__name__], tolerance, str(cell))
    reference = next(
        other
        for other in LOADERS
        if (kind.__name__, other) not in REFUSALS and (kind.__name__, other) not in ROUTE_ANSWERS
    )
    reference_outcome = _without_known_difference(
        (kind.__name__, reference), loaded.outcome(kind, reference)
    )
    _assert_close(outcome, reference_outcome, tolerance, f"{cell} versus {reference}")


# -- Requested covariates ------------------------------------------------------


# Definitions declare seven pre-period days, so revenue's own pre-period total is its
# covariate: units bought 3 + i % 4 on 2025-01-05 (sum 25, squares 111), with revenue
# cross products 106 in control and 122 in treatment. Frame and moments loaders
# declare no covariate for revenue and answer without one.
def _pre_period(cross: int, total_y: int) -> dict[str, object]:
    return {
        "role": "covariate",
        "mean": 25 / 6,
        "m2": 111 - 25**2 / 6,
        "c_y": cross - 25 * total_y / 6,
        "c_den": None,
    }


COVARIATE_MOMENTS: dict[str, object] = {
    "definitions": {
        "control": _m(6, 27, 191, x=_pre_period(106, 27)),
        "treatment": _m(6, 31, 249, x=_pre_period(122, 31)),
    },
    "unit_day_artifact": Refused("artifact.extension.missing", {}),
    "frame_summary": _TOTALS["revenue"],
    "frame_panel": _TOTALS["revenue"],
    "moments": _TOTALS["revenue"],
    "switchback_frame": Refused(*_grain(_SWITCHBACK, "total", ())),
}

# Known defect, tracked as "Definitions returns covariate moments without
# include_covariate": definitions attach the pre-period covariate to total moments that
# did not request it. Pinned exactly, then removed before any other comparison.
_UNREQUESTED_X = {
    "revenue": {"control": _pre_period(106, 27), "treatment": _pre_period(122, 31)},
    "purchase_rate": {
        arm: {"role": "covariate", "mean": 1.0, "m2": 0.0, "c_y": 0.0, "c_den": None}
        for arm in _ARMS
    },
    "rps": {
        "control": {**_pre_period(106, 27), "c_den": 0.0},
        "treatment": {**_pre_period(122, 31), "c_den": 0.0},
    },
}
KNOWN_DIFFERENCES: dict[tuple[str, str], object] = {
    ("MomentsRequest", "definitions"): _UNREQUESTED_X,
    ("EvidenceScope", "definitions"): _UNREQUESTED_X,
}


def _without_known_difference(cell: tuple[str, str], outcome: Any) -> Any:
    known = KNOWN_DIFFERENCES.get(cell)
    if known is None or isinstance(outcome, Refused):
        return outcome
    observed = {
        metric: {arm: row["x"] for arm, row in rows.items()} for metric, rows in outcome.items()
    }
    _assert_close(observed, known, _MOMENTS_TOLERANCE, f"{cell} known difference")
    return {
        metric: {arm: {**row, "x": None} for arm, row in rows.items()}
        for metric, rows in outcome.items()
    }


@pytest.mark.parametrize("loader", LOADERS)
def test_requested_covariate_moments(loader, loaded):
    source = loaded.sources[loader]
    try:
        outcome = _moment_rows(
            _ask(source, MomentsRequest(source.context.metrics[0], include_covariate=True))
        )
    except CodedError as refusal:
        outcome = Refused(refusal.code, _context(refusal))
    expected = COVARIATE_MOMENTS[loader]
    if isinstance(expected, Refused):
        assert outcome == expected
    else:
        _assert_close(outcome, expected, _MOMENTS_TOLERANCE, loader)


# A loader reduces each requested metric under its resolved configuration, so a slice
# request without them is refused alike before any read; a loader that serves no
# breakout refuses the dimension first.
@pytest.mark.parametrize("loader", LOADERS)
def test_breakout_slice_request_without_metric_configurations_is_refused(loader, loaded):
    source = loaded.sources[loader]
    request = BreakoutSlicesRequest(breakouts=(_STORE,), metrics=tuple(source.context.metrics))
    with pytest.raises(CodedError) as refused:
        _ask(source, request)
    code, context = REFUSALS.get(
        ("BreakoutSlicesRequest", loader),
        ("facade.analysis_config.metric_no_resolved", {"metric_name": "revenue"}),
    )
    assert Refused(refused.value.code, _context(refused.value)) == Refused(code, context)


# -- Declared capabilities this fixture cannot exercise ------------------------


def _compliance(summary: Any) -> dict[str, object]:
    return {
        "cohort": summary.cohort,
        "window_days": summary.window_days,
        "one_sided": summary.one_sided,
        "cluster": summary.cluster,
        "as_of": None if summary.as_of is None else str(summary.as_of),
        "arms": {
            arm.group_id: {"n_units": arm.n_units, "uptake_total": arm.uptake_total}
            for arm in summary.arms
        },
    }


def _checkpoint(snapshot: Any) -> dict[str, object]:
    return {
        "finalized": snapshot.finalized,
        "population": snapshot.population,
        "assignment_counts": dict(snapshot.assignment_counts),
        "units": {record.unit_id: record.group_id for record in snapshot.records},
        "states": {
            (state.metric, state.group_id): {
                "n": state.n,
                "successes": state.successes,
                "mean": tuple(state.mean),
                "scatter": tuple(tuple(row) for row in state.scatter),
            }
            for state in snapshot.states
        },
    }


WITNESSES: dict[type, tuple[str, Callable[[Any], object]]] = {
    ComplianceRequest: (
        "encouragement_panel",
        lambda source: _compliance(_ask(source, ComplianceRequest(design=source.context.design))),
    ),
    ComplianceSeriesRequest: (
        "encouragement_panel",
        lambda source: {
            str(day): _compliance(summary)
            for day, summary in _ask(source, ComplianceSeriesRequest(design=source.context.design))
        },
    ),
    SequentialCaptureRequest: (
        "registered_summary",
        lambda source: _checkpoint(_ask(source, SequentialCaptureRequest(finalized=True))),
    ),
    SequentialSnapshotRequest: (
        "registered_summary",
        lambda source: _checkpoint(_ask(source, SequentialSnapshotRequest())),
    ),
}


def _uptake(as_of: str | None) -> dict[str, object]:
    # Two units per arm on 2026-01-01 and 01-02; only treatment unit 0 clicks.
    return {
        "cohort": "clicked",
        "window_days": None,
        "one_sided": False,
        "cluster": None,
        "as_of": as_of,
        "arms": {
            "control": {"n_units": 2, "uptake_total": 0.0},
            "treatment": {"n_units": 2, "uptake_total": 1.0},
        },
    }


# The registered summary's units u1..u4 (control) earn 1..4 and u5..u8 (treatment)
# earn 4..7: sums 10 and 22, each with scatter 5.
_CHECKPOINT = {
    "finalized": True,
    "population": "assigned",
    "assignment_counts": {"control": 4, "treatment": 4},
    "units": {f"u{index}": "control" if index <= 4 else "treatment" for index in range(1, 9)},
    "states": {
        ("revenue", "control"): {
            "n": 4,
            "successes": None,
            "mean": (Fraction(10, 4),),
            "scatter": ((Fraction(5),),),
        },
        ("revenue", "treatment"): {
            "n": 4,
            "successes": None,
            "mean": (Fraction(22, 4),),
            "scatter": ((Fraction(5),),),
        },
    },
}
WITNESS_EXPECTED: dict[str, object] = {
    "ComplianceRequest": _uptake(None),
    "ComplianceSeriesRequest": {day: _uptake(day) for day in ("2026-01-01", "2026-01-02")},
    "SequentialCaptureRequest": _CHECKPOINT,
    "SequentialSnapshotRequest": _CHECKPOINT,
}


@pytest.fixture(scope="module")
def witnesses() -> Iterator[Mapping[str, Any]]:
    """An encouragement panel and a summary built with a registered checkpoint."""
    built = {
        "encouragement_panel": encouragement_panel_analysis(),
        "registered_summary": _registered_unit_summary_analysis(),
    }
    try:
        yield MappingProxyType({name: bound_source(pair[1]) for name, pair in built.items()})
    finally:
        for _, resolver in built.values():
            resolver.close()


@pytest.mark.parametrize("kind", tuple(WITNESSES), ids=lambda kind: kind.__name__)
def test_declared_kind_answers_on_its_witness(kind, witnesses):
    name, acquire = WITNESSES[kind]
    source = witnesses[name]
    assert kind in _declared_kinds(source.evidence), (name, source.evidence)
    _assert_close(acquire(source), WITNESS_EXPECTED[kind.__name__], _VALUE_TOLERANCE, name)


# Seven units per arm, the fewest that give the second store two units in each arm:
# unit i buys 5 + i % 5 (one more in treatment) inside its window unless i % 3 == 0.
# Units 5 and 6 shop at s1, and so does treatment unit 3, so the arms' store mixes
# differ and the adjustment moves both arm means.
_FIT_REVENUE = {
    "control": (0.0, 6.0, 7.0, 0.0, 9.0, 5.0, 0.0),
    "treatment": (0.0, 7.0, 8.0, 0.0, 10.0, 6.0, 0.0),
}
_FIT_S1_UNITS = {"control": (5, 6), "treatment": (3, 5, 6)}
# Declared on the plan so the artifact publishes the store levels.
_DECLARED_STORE_CUPED = {
    "name": "mixed",
    "variance_reduction": "cuped_multicovariate",
    "covariates": [
        {"name": "store", "kind": "categorical", "property": "store", "source": "events"}
    ],
}


def _least_squares_fit() -> dict[str, object]:
    """The store-adjusted fit, solved by numpy least squares from the unit rows.

    Arm intercepts and one slope on the s1 indicator (the modal store s0 is the
    reference), centred on its pooled mean, with each arm's units weighted
    1 / (n_a (n_a - 1)): the intercepts are the adjusted arm means. Uncertainty is
    HC2 with Bell-McCaffrey degrees of freedom. Each arm mean also carries the pooled
    mean's sampling error, slope * (x_i - x_bar of i's arm) / n per unit, as the
    design fixes the arm sizes; it cancels in the difference of means.
    """
    arm = np.repeat([0, 1], 7)
    y = np.array([*_FIT_REVENUE["control"], *_FIT_REVENUE["treatment"]])
    x = np.array([float(i in _FIT_S1_UNITS[name]) for name in _ARMS for i in range(7)])
    n = np.bincount(arm)
    root_weight = np.sqrt(1 / (n * (n - 1)))[arm]
    anchor = x.mean()
    design = np.column_stack([arm == 0, arm == 1, x - anchor]) * root_weight[:, None]
    coefficients, *_ = np.linalg.lstsq(design, y * root_weight, rcond=None)
    control, treatment, slope = coefficients
    bread = np.linalg.inv(design.T @ design)
    hat = design @ bread @ design.T
    leverage = np.diag(hat)
    residual = (y * root_weight - design @ coefficients) / np.sqrt(1 - leverage)
    within = x - np.array([x[arm == index].mean() for index in (0, 1)])[arm]

    def score(gradient: np.ndarray) -> np.ndarray:
        influence = design @ bread @ gradient
        return influence * residual + gradient.sum() / len(y) * slope * within

    def projection(gradient: np.ndarray) -> tuple[float, float]:
        influence = design @ bread @ gradient
        spread = (np.eye(len(y)) - hat) * (influence / np.sqrt(1 - leverage))
        moment = spread.T @ spread
        dof = np.trace(moment) ** 2 / np.trace(moment @ moment)
        return np.linalg.norm(score(gradient)), dof

    arm_scores = [score(np.eye(3)[index]) for index in (0, 1)]
    return {
        "origin": "warehouse",
        "withheld": (),
        "arm_order": _ARMS,
        "n_by_arm": (7, 7),
        "levels": (("s0", "s1"),),
        "references": ("s0",),
        "anchor": (anchor,),
        "theta": (slope,),
        "adjusted_means": (control, treatment),
        "covariance": tuple(tuple(left @ right for right in arm_scores) for left in arm_scores),
        "leverage": (leverage.max(), leverage.mean(), leverage.sum()),
        "absolute": (treatment - control, *projection(np.array([-1.0, 1.0, 0.0]))),
        "relative": (
            np.log(treatment / control),
            *projection(np.array([-1 / control, 1 / treatment, 0.0])),
        ),
    }


def _fit_evidence(source: Any) -> dict[str, object]:
    answer = _store_fit(source)
    fit = answer.envelope.fit
    absolute, relative = fit.absolute("treatment"), fit.relative("treatment")
    return {
        "origin": answer.origin,
        "withheld": answer.envelope.withheld,
        "arm_order": fit.arm_order,
        "n_by_arm": fit.n_by_arm,
        "levels": fit.layout.levels,
        "references": fit.layout.references,
        "anchor": fit.anchor,
        "theta": fit.theta,
        "adjusted_means": fit.adjusted_means,
        "covariance": fit.covariance,
        "leverage": (fit.max_leverage, fit.mean_leverage, fit.leverage_trace),
        "absolute": (absolute.estimate, absolute.se, absolute.dof),
        "relative": (relative.estimate, relative.se, relative.dof),
    }


@pytest.fixture(scope="module")
def store_fit_sources() -> Iterator[Mapping[str, Any]]:
    """Definitions and an artifact published with the store covariate, over the
    fixture above."""
    rows = [
        {**row, "store_id": "s1"} if row["user_id"] == "t3" else row
        for row in dataset.event_rows(n_per_arm=7, n_stores=2)
    ]
    plan = AnalysisPlan(
        primary=ExperimentMetric(metric="revenue", sensitivity_methods=(_DECLARED_STORE_CUPED,))
    )
    definitions = Definitions.model_validate(dataset.definitions_dict(plan=plan))
    built: dict[str, tuple[Any, Any]] = {}
    try:
        for name in ("definitions", "unit_day_artifact"):
            con = dataset.duckdb_connection(rows)
            native = make_binding(con, definitions, experiment="exp")
            built[name] = (
                _track_connection(native, con)
                if name == "definitions"
                else _publish_and_adopt(con, native, kinds=("unit_covariate_level",))
            )
        yield MappingProxyType({name: bound_source(pair[1]) for name, pair in built.items()})
    finally:
        for pair in built.values():
            _close_parity_binding(pair)


@pytest.mark.parametrize("loader", ("definitions", "unit_day_artifact"))
def test_method_fit_matches_least_squares_on_unit_rows(loader, store_fit_sources):
    source = store_fit_sources[loader]
    assert MethodFitRequest in _declared_kinds(source.evidence), (loader, source.evidence)
    _assert_close(_fit_evidence(source), _least_squares_fit(), _VALUE_TOLERANCE, loader)
