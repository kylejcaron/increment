"""The enumerated parity matrix: axes, cells, and the disposition of every cell.

``tests/test_parity_matrix.py`` runs every cell of the cross product

    metric x view x option x day_boundary x missing

through all six ingress constructors and checks the result against the
``Disposition`` that ``classify`` returns for it. This module is the single
machine-readable source of those dispositions; nothing else restates them, and
``tests/parity_harness/COVERAGE.md`` only counts them.

A cell's disposition is one ``Verdict`` per ingress and leg (a day-axis view has two legs,
the value series and the lift, that refuse independently): what it does (``Runs``,
``Refuses(code)`` or ``StructuralAbsence``), the status that explains it, a reason
and an authority independent of the refusal itself. The statuses are

``supported``            runs; a matched-arm ingress also agrees with every other running
                         matched-arm ingress within the runner's 1e-9 relative tolerance, while
                         ``from_switchback_panel`` (a different estimand) runs on its own
``source_limited``       the ingress cannot supply the input (``SOURCE:``)
``construction_limited`` the estimator or readout is not defined for the combination
                         (``CONSTRUCTION:``/``COMBINATION:``)
``not_expressible``      the axis value cannot be declared on that ingress; the outcome is
                         the declaration-time refusal
``unfinished``           supportable or unifiable but not implemented; carries a tracker
                         (``UNFINISHED(<ref>):``)
``unsound``              mathematically invalid; carries a derivation

The label in a reason is documentation; the runner does not enforce it. The cell's own
status is derived (see ``Disposition.status``).

Rules
-----
Per ingress an ordered ``Box`` tuple names the outcome of each (metric, view, option,
missing) region; the first matching box wins and no box matches only what runs. The order
is the order in which that ingress's validators look at a request, so an earlier box can
shadow a later one. ``day_boundary`` never selects a box: it moves day labels and window
days, never whether a request is accepted, and the executor runs both boundaries.

An ingress that runs while another refuses is valid only where the refuser is
``source_limited``/``not_expressible``/``construction_limited``/``unfinished`` with a
reason and an authority. Construction-limited refusals of one hazard must carry one code:
within a declaration surface (frame ``MetricSpec`` versus warehouse ``Definitions``), and
across every ingress for a request-stage refusal. ``check_disposition`` enforces both, so a
divergence is either explained as ``unfinished`` with a tracker or fails.

No I/O at import; builders live in ``matrix_cases``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import ValidationError

from .cases import CONSTRUCTORS

INGRESSES = CONSTRUCTORS

BASE_METRICS = ("mean", "conversion", "ratio", "retention", "quantile", "total", "active")
METRICS = (*BASE_METRICS, *(f"windowed_{base}" for base in BASE_METRICS))
VIEWS = ("run", "breakout", "daily", "asof")
OPTIONS = (
    "none",
    "cuped",
    "winsor_fixed",
    "winsor_percentile",
    "cluster",
    "sequential",
    "observational",
    "ni_margin",
)
DAY_BOUNDARIES = ("utc", "fixed_offset")
MISSING = ("error", "zero", "drop", "impute")

Status = Literal[
    "supported",
    "source_limited",
    "construction_limited",
    "not_expressible",
    "unfinished",
    "unsound",
]
Stage = Literal["declaration", "request"]


@dataclass(frozen=True)
class Cell:
    metric: str
    view: str
    option: str
    day_boundary: str
    missing: str

    @property
    def id(self) -> str:
        return "-".join((self.metric, self.view, self.option, self.day_boundary, self.missing))

    @property
    def windowed(self) -> bool:
        return self.metric.startswith("windowed_")

    @property
    def base(self) -> str:
        return self.metric.removeprefix("windowed_")


def iter_cells() -> Iterator[Cell]:
    """Every cell of the cross product, in a fixed order."""
    for metric, view, option, boundary, missing in itertools.product(
        METRICS, VIEWS, OPTIONS, DAY_BOUNDARIES, MISSING
    ):
        yield Cell(metric, view, option, boundary, missing)


@dataclass(frozen=True)
class Runs:
    """The ingress produces rows. A matched-arm ingress's rows are compared with every other
    running matched-arm ingress; ``from_switchback_panel`` (a different estimand) is run on its
    own and its rows are only required to exist."""


@dataclass(frozen=True)
class Refuses:
    """The ingress raises ``CodedError`` with exactly this code."""

    code: str


@dataclass(frozen=True)
class StructuralAbsence:
    """The constructor or schema cannot express the request.

    The runner attempts the declaration with the unsupported field or keyword and expects
    exactly ``error`` (``TypeError`` for a keyword, pydantic's ``ValidationError`` for a
    schema field) naming ``field`` as the thing it cannot express: the validation error
    must locate ``field``, and a keyword must be absent from the constructor's signature.
    No refusal code exists to record.
    """

    fact: str
    field: str
    error: type[Exception] = TypeError


Outcome = Runs | Refuses | StructuralAbsence


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    status: Status
    reason: str
    authority: str
    stage: Stage = "request"
    tracker: str | None = None
    derivation: str | None = None
    hazard: str | None = None

    def __post_init__(self) -> None:
        if not self.reason or not self.authority:
            raise ValueError("a verdict needs a reason and an authority")
        if self.status == "unfinished" and not self.tracker:
            raise ValueError("an unfinished verdict needs a tracker")
        if self.status == "unsound" and not self.derivation:
            raise ValueError("an unsound verdict needs a cited derivation")
        if isinstance(self.outcome, Runs) != (self.status == "supported"):
            raise ValueError("only a running ingress is supported")


def methods(cell: Cell) -> tuple[str, ...]:
    """The independently executed legs of a cell.

    ``run`` and ``breakout`` read one method (``rows``). A day-axis view reads two that
    refuse independently: the value series (``values``: ``run_daily``/``run_asof``) and the
    lift (``lift``: ``run_daily_lift``/``run_asof_lift``).
    """
    return ("rows",) if cell.view in ("run", "breakout") else ("values", "lift")


@dataclass(frozen=True)
class Disposition:
    """Per leg (``methods``) and ingress, the verdict."""

    legs: Mapping[str, Mapping[str, Verdict]]

    def __post_init__(self) -> None:
        for method, verdicts in self.legs.items():
            if set(verdicts) != set(INGRESSES):
                raise ValueError(f"{method}: verdicts must cover exactly the six ingresses")

    def outcomes(self, method: str) -> dict[str, Outcome]:
        return {name: v.outcome for name, v in self.legs[method].items()}

    @property
    def all_verdicts(self) -> list[Verdict]:
        return [v for verdicts in self.legs.values() for v in verdicts.values()]

    @property
    def status(self) -> Status:
        """The cell's status: the most significant verdict over every leg and ingress.

        ``unfinished`` or ``unsound`` anywhere wins; else ``supported`` when anything
        runs; else ``not_expressible`` when every refusal is a declaration the axis value
        cannot make; else ``construction_limited`` when any refusal is, else
        ``source_limited``.
        """
        statuses = {v.status for v in self.all_verdicts}
        for ranked in ("unfinished", "unsound", "supported"):
            if ranked in statuses:
                return ranked
        if statuses == {"not_expressible"}:
            return "not_expressible"
        if "construction_limited" in statuses:
            return "construction_limited"
        return "source_limited"

    @property
    def trackers(self) -> tuple[str, ...]:
        return tuple(sorted({v.tracker for v in self.all_verdicts if v.tracker}))


@dataclass(frozen=True)
class Box:
    """A region of (metric, view, option, missing); ``None`` on an axis matches every value.

    ``why`` is the observed signature the region records (``RUNS``, ``REF:<code>[#reason]``
    or ``EXC:<type>[#field]``); ``_SPECS`` explains it.
    """

    why: str
    metric: tuple[str, ...] | None = None
    view: tuple[str, ...] | None = None
    option: tuple[str, ...] | None = None
    missing: tuple[str, ...] | None = None
    method: str = "rows"

    def matches(self, cell: Cell, method: str) -> bool:
        return (
            self.method == method
            and (self.metric is None or cell.metric in self.metric)
            and (self.view is None or cell.view in self.view)
            and (self.option is None or cell.option in self.option)
            and (self.missing is None or cell.missing in self.missing)
        )


def box(
    why: str,
    metric: str = "*",
    view: str = "*",
    option: str = "*",
    missing: str = "*",
    method: str = "rows",
) -> Box:
    """A ``Box`` from space-separated axis values; ``*`` matches every value."""

    def axis(text: str) -> tuple[str, ...] | None:
        return None if text == "*" else tuple(text.split())

    return Box(why, axis(metric), axis(view), axis(option), axis(missing), method)


@dataclass(frozen=True)
class Spec:
    """Why a region has the outcome it has: status, stage, reason and authority."""

    status: Status
    reason: str
    authority: str
    stage: Stage = "request"
    tracker: str | None = None
    fact: str = ""


_FRAMES = ("from_unit_summary", "from_unit_panel", "from_moments", "from_switchback_panel")
_WAREHOUSE = ("from_definitions", "from_unit_day_artifact")

_CATALOG = "tests/compatibility_catalog.py::MATRIX"
_LIMITATIONS = 'docs/limitations.md "What runs where"'
_METRIC_SPECS = "increment/_metric_specs.py"
_DAY_AXIS = "increment/_day_axis.py"
_RUN_ASOF_LIFT = "Analysis.run_asof_lift docstring"
_RUN_DAILY_LIFT = "Analysis.run_daily_lift docstring"
_PMP3 = "pmp3 policy-home ledger (kata comment on pmp3, S15)"
_RUNNER = "tests/parity_harness/runner.py::_TOLERANCE (1e-9 relative)"

# Tracker ids of the unfinished cells; each cell carries the code it raises now.
_T_QUANTILE_DAY_AXIS = "0f6d"
_T_WINDOWED_QUANTILE = "6z2f"
_T_ARTIFACT_QUANTILE_CUPED = "66mg"
_T_SEQUENTIAL_UNBOUNDED = "1cr4"
_T_QUANTILE_ONE_SIDED = "r3bg"

_SPECS: dict[str, Spec] = {
    # -- declaration surfaces: nothing is built from these axis values ---------------------
    "REF:frame.metric_unknown_type": Spec(
        "not_expressible",
        "total and active are report-layer metric types: a frame names no such type",
        f"{_METRIC_SPECS}::_as_type; {_CATALOG} _NA_REPORT_LAYER",
        "declaration",
    ),
    "REF:definition.invalid#report_only": Spec(
        "not_expressible",
        "a report-only metric (total, active) has no per-unit outcome, so an experiment "
        "cannot name it",
        "increment/semantics/models.py::Definitions consistency check; "
        f"{_CATALOG} _NA_REPORT_LAYER",
        "declaration",
    ),
    "REF:definition.total.metric_window_days": Spec(
        "not_expressible",
        "window_days is meaningless for a report-only metric: the calendar period is the window",
        "increment/semantics/models.py::TotalMetric",
        "declaration",
    ),
    "REF:definition.active.metric_window_days": Spec(
        "not_expressible",
        "window_days is meaningless for a report-only metric: the calendar period is the window",
        "increment/semantics/models.py::ActiveMetric",
        "declaration",
    ),
    "REF:definition.retention.metric_window_days": Spec(
        "not_expressible",
        "a retention metric carries both edges of its observation band in threshold_days; "
        "window_days is not a retention field",
        "increment/semantics/models.py::RetentionMetric",
        "declaration",
    ),
    "EXC:ValidationError#winsorization": Spec(
        "not_expressible",
        "the Definitions schema declares winsorization on type: mean only",
        "increment/semantics/models.py::MeanMetric (the only metric model with the field)",
        "declaration",
        fact="a non-mean Definitions metric has no winsorization field",
    ),
    "EXC:ValidationError#missing": Spec(
        "not_expressible",
        "a warehouse Metric cannot express a missing-value policy: absent events are zero by "
        "construction of the event log",
        f"{_LIMITATIONS}: MetricSpec(missing=...) row, warehouse routes",
        "declaration",
        fact="no Definitions metric has a missing field",
    ),
    "EXC:TypeError#cluster": Spec(
        "source_limited",
        "from_switchback_panel takes no cluster= parameter: a switchback contrast has no "
        "arm-cluster shape",
        f"Analysis.from_switchback_panel signature; {_LIMITATIONS} clustered rows",
        "declaration",
        fact="from_switchback_panel has no cluster= keyword",
    ),
    "REF:frame.metric.winsorization_applies_type": Spec(
        "construction_limited",
        "winsorization bounds a per-unit outcome and is defined for a mean only",
        f"{_METRIC_SPECS}::MetricSpec._check_winsorization; docs/guides/metric-types.md",
        "declaration",
    ),
    "REF:frame.metric.cuped_does_apply": Spec(
        "construction_limited",
        "a quantile is not a mean of per-unit values: there is no per-unit residual for a "
        "covariate slope to act on",
        f"{_CATALOG}['cuped']['quantile']; {_METRIC_SPECS}::MetricSpec._check_methods",
        "declaration",
    ),
    "REF:frame.metric.missing_impute": Spec(
        "construction_limited",
        "filling an outcome with its pooled mean shrinks variance and biases the estimate",
        f"{_LIMITATIONS}: impute.pooled_mean_outcome row; {_METRIC_SPECS}::MetricSpec",
        "declaration",
    ),
    "REF:frame.metric.window_days_supported": Spec(
        "unfinished",
        "a windowed quantile is declared by the warehouse model and read from per-unit "
        "totals, yet refused on the frame path with no stated statistical reason",
        f"{_LIMITATIONS}: quantile row; {_METRIC_SPECS}::MetricSpec._check_windowing",
        "declaration",
        tracker=_T_WINDOWED_QUANTILE,
    ),
    # -- source-limited: the ingress cannot supply the input ------------------------------
    "REF:frame.missing_policy.panel_drop": Spec(
        "source_limited",
        "a panel is densified to a unit x day spine of zero-filled cells, so a dropped "
        "(unit, day) row would silently mean zero",
        "increment/_frame_validation.py::_PANEL_MISSING_DROP",
        "declaration",
    ),
    "REF:source.frame.constructor#window": Spec(
        "source_limited",
        "a one-row-per-unit summary carries no dates to window against",
        "increment/_frame_validation.py::CAPABILITY_TABLE['window_days']; "
        "MetricSpec docstring (from_unit_summary carries no dates)",
        "declaration",
    ),
    "REF:source.frame.constructor#retention": Spec(
        "source_limited",
        "a retention band needs an exposure date to count days from; a summary has none",
        f"increment/_frame_validation.py::CAPABILITY_TABLE['type=retention']; {_CATALOG} _SEAM",
        "declaration",
    ),
    "REF:facade.analysis.no_definitions": Spec(
        "source_limited",
        "the source has no day axis: day-axis methods need a native or panel source",
        "Analysis.from_unit_summary and Analysis.from_moments docstrings (day-axis methods "
        "raise CapabilityError)",
    ),
    "REF:facade.analysis.operation": Spec(
        "source_limited",
        "the source carries no breakout catalog: run_breakout needs a native or panel source",
        "Analysis.from_unit_summary and Analysis.from_moments docstrings; "
        f"{_LIMITATIONS}: breakout rows",
    ),
    "REF:facade.analysis.no_definitions#observational_quantile": Spec(
        "source_limited",
        "a scalar-moments cube replayed with a declared observational quantile has no day axis: "
        "the day-axis gate refuses before any estimator reads the cube, so the quantile "
        "refusal run() raises is not reached on this view",
        "Analysis.from_moments docstring (day-axis methods raise CapabilityError); measured "
        "on a real scalar-moments source (matrix_cases.py::_Ingress._scalar_moments_producer)",
    ),
    "REF:facade.analysis.operation#observational_quantile": Spec(
        "source_limited",
        "a scalar-moments cube replayed with a declared observational quantile carries no "
        "breakout catalog: run_breakout refuses at the source gate before any estimator "
        "reads the cube, so the quantile refusal run() raises is not reached on this view",
        "Analysis.from_moments docstring; "
        f"{_LIMITATIONS}: breakout rows; measured on a real scalar-moments source "
        "(matrix_cases.py::_Ingress._scalar_moments_producer)",
    ),
    "REF:source.moments.grain": Spec(
        "source_limited",
        "a moments cube offers only the grains it was exported with: total, not asof",
        "increment/sources.py::MomentsSource (source.moments.grain reports the offered set); "
        f"{_LIMITATIONS}: checkpoint replay",
    ),
    "REF:source.moments.cluster_grain": Spec(
        "source_limited",
        "the moments wire format has no cluster marker, so clustered inference cannot be "
        "transported",
        f"{_LIMITATIONS}: clustered rows (source.moments.cluster_grain)",
        "declaration",
    ),
    "REF:source.frame_panel.cluster_grain": Spec(
        "source_limited",
        "a per-day panel collapses to one row per unit before clustering could apply; "
        "the collapse is ambiguous on this shape",
        f"{_LIMITATIONS}: clustered rows (from_unit_panel(cluster=...) refuses)",
        "declaration",
    ),
    "REF:source.moments.covariate_unavailable": Spec(
        "source_limited",
        "a moments cube holds no per-unit rows to attach an adjustment covariate to",
        f"{_LIMITATIONS}: observational rows (source.moments.covariate_unavailable)",
    ),
    "REF:source.frame.unit_frame_panel": Spec(
        "source_limited",
        "a windowed or retention panel has no per-unit collapse to serve a unit frame",
        f"{_LIMITATIONS}: panel covariate paragraph; increment/_frame_validation.py",
    ),
    "REF:source.frame.retention_daily": Spec(
        "source_limited",
        "a frame panel has no independent per-day (cohort) reading of retention",
        f"{_PMP3}: source-limited frame retention_daily; Analysis.run_daily_lift docstring",
    ),
    "REF:estimation.winsor.raw_state_required": Spec(
        "source_limited",
        "percentile winsorization needs the exact pre-winsor unit outcomes, which this "
        "source does not retain",
        f"{_PMP3}: source-limited estimation.winsor.raw_state_required; "
        "docs/guides/metric-types.md",
    ),
    "REF:estimation.cuped.arm_no_covariate": Spec(
        "source_limited",
        "a dataframe panel carries no pre-period covariate moments on the day axis",
        f"{_RUN_ASOF_LIFT} (CUPED remains refused for dataframe-panel sources)",
    ),
    "REF:frame.validation.from_unit_panel": Spec(
        "construction_limited",
        "a windowed or retention metric has no per-unit collapse a pre-period covariate "
        "can attach to; take the pre-period value into from_unit_summary",
        f"{_LIMITATIONS}: panel covariate paragraph (CONSTRUCTION)",
        "declaration",
    ),
    # -- construction-limited at request time: one hazard, one code -----------------------
    "REF:estimation.adjust_common.supported_ratio_metric": Spec(
        "construction_limited",
        "inverse-propensity adjustment is not defined for a ratio metric",
        f"{_CATALOG}['observational']['ratio']",
    ),
    "REF:arm.metric.quantile_cluster": Spec(
        "construction_limited",
        "a quantile does not decompose over cluster-grain moments",
        f"{_CATALOG}['cluster']['quantile']; increment/compatibility.py ARM_COMPATIBILITY_REFUSALS",
    ),
    "REF:arm.metric.quantile_cuped": Spec(
        "construction_limited",
        "a quantile has no mean to adjust",
        f"{_CATALOG}['cuped']['quantile']; increment/compatibility.py ARM_COMPATIBILITY_REFUSALS",
    ),
    "REF:source.frame.cluster_capability": Spec(
        "construction_limited",
        "a quantile does not decompose over cluster-grain moments",
        f"{_CATALOG}['cluster']['quantile']",
        "declaration",
    ),
    "REF:facade.analysis.clustered_day_axis": Spec(
        "construction_limited",
        "a declared cluster is total-grain only: cluster-robust inference has no per-day form",
        f"{_DAY_AXIS}::_CLUSTERED_DAY_AXIS; {_LIMITATIONS}: clustered rows",
    ),
    "REF:definition.invalid#cluster": Spec(
        "construction_limited",
        "cluster-robust inference is total-grain only, so a declared breakout could never "
        "be served",
        "increment/semantics/models.py::Experiment consistency check; "
        f"{_DAY_AXIS}::_CLUSTERED_DAY_AXIS",
        "declaration",
    ),
    "REF:facade.analysis.observational_day_axis": Spec(
        "construction_limited",
        "confounded day-axis contrasts are refused, not silently emitted",
        f"{_RUN_ASOF_LIFT} (observational designs refuse); {_DAY_AXIS}",
    ),
    "REF:readout.view.observational": Spec(
        "construction_limited",
        "confounded per-segment contrasts are refused, not silently emitted",
        "Analysis.run_breakout docstring (an observational panel refuses this method by name)",
    ),
    "REF:readout.margin.breakout": Spec(
        "construction_limited",
        "breakout rows are tested two-sided against zero; a per-segment shifted null is "
        "not built, so the guardrail read is run()",
        "Analysis.run_breakout docstring; increment/estimation/engine.py::validate_readout_engine",
    ),
    "REF:readout.metric.percentile_winsorization": Spec(
        "construction_limited",
        "a breakout read of a percentile-winsorized metric needs raw independent units and a "
        "supported two-sided winsor inference specification",
        "docs/guides/metric-types.md; increment/estimation/engine.py::validate_readout_engine",
    ),
    "REF:breakout.metric.daily_winsorization": Spec(
        "construction_limited",
        "winsorization is a total-grain construction: run_daily and run_asof refuse it",
        f"{_PMP3}: daily winsorization (both codes kept); "
        "increment/breakout/estimates.py::reject_winsorized_day_axis",
    ),
    "REF:readout.inference.disjoint_slices": Spec(
        "construction_limited",
        "per-day and per-cohort slices are disjoint, so a sequential guarantee cannot apply",
        f"{_RUN_DAILY_LIFT} (use run_asof_lift for cumulative monitoring)",
    ),
    "REF:readout.metric.quantile_breakout": Spec(
        "construction_limited",
        "quantiles do not decompose over segment moments",
        f"{_CATALOG}['breakout']['quantile']",
    ),
    "REF:sequential.route.unsupported#unbounded": Spec(
        "construction_limited",
        "every observation window must be bounded by the common registered reveal window",
        f"{_LIMITATIONS}: sequential rows; AGENTS.md (a sequential transform must be "
        "predictable when each observation arrives)",
    ),
    "REF:sequential.route.unsupported#panel_unbounded": Spec(
        "construction_limited",
        "panel metrics need bounded windows covered by the common joint-reveal window",
        f"{_LIMITATIONS}: sequential rows; Analysis.from_unit_panel docstring",
        "declaration",
    ),
    "REF:sequential.route.unsupported#metric_type": Spec(
        "construction_limited",
        "automatic sequential inference needs a mean, ratio, conversion or retention metric: "
        "a quantile has no matching sequential sampling proof",
        f"{_CATALOG}['sequential']['quantile']",
        "declaration",
    ),
    "REF:sequential.route.unsupported#drop": Spec(
        "construction_limited",
        "outcome-dependent missing-row deletion breaks the joint reveal",
        "AGENTS.md (a sequential transform must be predictable when each observation arrives)",
        "declaration",
    ),
    "REF:sequential.route.unsupported#breakout": Spec(
        "construction_limited",
        "a breakout readout under sequential inference needs one registered segment "
        "dimension; none is declared",
        f"{_LIMITATIONS}: registered segmented sequential family",
    ),
    "REF:readout.metric.quantile_grain": Spec(
        "construction_limited",
        "quantiles do not decompose into per-day moments",
        f"{_CATALOG}['daily_asof']['quantile']",
    ),
    # -- unfinished: code divergence or an unimplemented route ----------------------------
    "REF:breakout.quantile": Spec(
        "unfinished",
        "the same quantile x day-axis hazard raises the breakout code here, not the catalog's "
        "readout.metric.quantile_grain",
        f"{_CATALOG}['daily_asof']['quantile'] declares readout.metric.quantile_grain",
        tracker=_T_QUANTILE_DAY_AXIS,
    ),
    "REF:query.builders.asof_group_summary_metric_type_not_implemented": Spec(
        "unfinished",
        "an internal not-implemented code surfaces for quantile x asof instead of the "
        "catalog's readout.metric.quantile_grain",
        f"{_CATALOG}['daily_asof']['quantile'] declares readout.metric.quantile_grain",
        tracker=_T_QUANTILE_DAY_AXIS,
    ),
    "REF:frame.asof.quantile_unsupported": Spec(
        "unfinished",
        "the panel raises its own quantile x asof code instead of the catalog's "
        "readout.metric.quantile_grain",
        f"{_CATALOG}['daily_asof']['quantile'] declares readout.metric.quantile_grain",
        tracker=_T_QUANTILE_DAY_AXIS,
    ),
    "REF:readout.observational.quantile#asof_source": Spec(
        "unfinished",
        "the artifact's descriptive as-of value path never reaches the observational readout "
        "seam; the source's observational-quantile guard raises the estimator refusal "
        "instead of the catalog's readout.metric.quantile_grain",
        f"{_CATALOG}['daily_asof']['quantile'] declares readout.metric.quantile_grain; "
        "increment/query/source.py::_ArtifactFacadeSource._refuse_observational_quantile",
        tracker=_T_QUANTILE_DAY_AXIS,
    ),
    "REF:artifact.extension.invalid": Spec(
        "unfinished",
        "publishing the CUPED extension for a quantile raises an artifact code instead of the "
        "arm-compatibility code the other routes raise",
        f"{_CATALOG}['cuped']['quantile']; increment/compatibility.py ARM_COMPATIBILITY_REFUSALS",
        tracker=_T_ARTIFACT_QUANTILE_CUPED,
    ),
    "REF:sequential.source.invalid": Spec(
        "unfinished",
        "an unbounded-window sequential request raises a source code here while the "
        "definitions and panel routes raise sequential.route.unsupported",
        f"{_LIMITATIONS}: sequential rows",
        tracker=_T_SEQUENTIAL_UNBOUNDED,
    ),
    "REF:readout.metric.quantile_alternative": Spec(
        "unfinished",
        "a one-sided alternative (a non-inferiority margin) is not supported for quantile "
        "metrics yet",
        "increment/estimation/engine.py::validate_readout_engine (message: 'yet')",
        tracker=_T_QUANTILE_ONE_SIDED,
    ),
    "REF:readout.observational.quantile": Spec(
        "construction_limited",
        "an observational design has no quantile estimator: the readout seam refuses a quantile "
        "metric before any source read, including one declared over a scalar-moments cube",
        "increment/estimation/_readout_refusals.py::refuse_observational_quantile (called from "
        "estimation/adjust.py::validate_readout_adjustment and estimate_ate); "
        f"{_CATALOG}['observational']['quantile']",
    ),
    # -- switchback: a different estimand, fixed-horizon contrasts only ------------------
    "REF:source.frame.switchback.identification": Spec(
        "source_limited",
        "a switchback frame identifies a Randomized design only: restricted, observational and "
        "encouragement designs are unsupported",
        f"{_LIMITATIONS}: observational rows (refused, SOURCE -- no design=)",
        "declaration",
    ),
    "REF:source.frame.switchback.metric#type": Spec(
        "source_limited",
        "switchback metrics must be a mean or a conversion",
        f"{_LIMITATIONS}: quantile rows (refused at the metric-type gate)",
        "declaration",
    ),
    "REF:source.frame.switchback.metric#method": Spec(
        "source_limited",
        "switchback procedures apply no decision-method variance reduction",
        f"{_LIMITATIONS}: the switchback panel supports neither sequential inference nor CUPED",
        "declaration",
    ),
    "REF:source.frame.switchback.metric#missing": Spec(
        "source_limited",
        "switchback aggregation applies no missing-value policy",
        "increment/switchback.py::_SWITCHBACK_METRIC",
        "declaration",
    ),
    "REF:source.frame.switchback.metric#window": Spec(
        "source_limited",
        "a switchback retained window is declared by the assignment, not by window_days",
        "increment/switchback.py::_SWITCHBACK_METRIC; Analysis.from_switchback_panel docstring",
        "declaration",
    ),
    "REF:source.frame.switchback.metric#winsor": Spec(
        "source_limited",
        "switchback aggregation applies no winsorization",
        "increment/switchback.py::_SWITCHBACK_METRIC",
        "declaration",
    ),
    "REF:source.frame.switchback.plan#inference": Spec(
        "source_limited",
        "switchback contrasts support fixed inference only",
        f"{_LIMITATIONS}: the switchback panel supports neither sequential inference nor CUPED",
        "declaration",
    ),
    "REF:source.frame.switchback.plan#margin": Spec(
        "source_limited",
        "switchback contrasts do not support relative margins",
        "increment/decision.py (contrast procedure compilation)",
        "declaration",
    ),
    "REF:facade.analysis.contrast_unavailable": Spec(
        "source_limited",
        "switchback contrast evidence has no breakout, daily or as-of view",
        "Analysis.from_switchback_panel docstring (one fixed-horizon contrast)",
    ),
}


def _outcome(why: str, spec: Spec) -> Outcome:
    if why == "RUNS":
        return Runs()
    kind, _, rest = why.partition(":")
    name, _, field = rest.partition("#")
    if kind == "REF":
        return Refuses(name)
    return StructuralAbsence(
        spec.fact, field, {"ValidationError": ValidationError, "TypeError": TypeError}[name]
    )


# The same observed signature is a different thing on one ingress.
_OVERRIDES: dict[tuple[str, str], Spec] = {
    ("from_moments", "REF:frame.validation.from_unit_panel"): Spec(
        "construction_limited",
        "a cube is exported by a producer; under a drop policy the only producer of a windowed "
        "or retention metric is the dataframe panel, which refuses a pre-period covariate (the "
        "warehouse producer that carries one declares no missing policy)",
        f"{_LIMITATIONS}: panel covariate paragraph (CONSTRUCTION)",
        "declaration",
    ),
    ("from_unit_day_artifact", "REF:frame.metric.window_days_supported"): Spec(
        "unfinished",
        "the artifact route reads a windowed quantile through the frame declaration and refuses "
        "it, while the definitions route that published it runs",
        f"{_LIMITATIONS}: quantile row; {_METRIC_SPECS}::MetricSpec._check_windowing",
        tracker=_T_WINDOWED_QUANTILE,
    ),
    ("from_moments", "REF:source.frame.quantile_no_moments"): Spec(
        "source_limited",
        "a quantile has no moments representation and a cube holds no per-unit rows to serve one",
        f"{_LIMITATIONS}: quantile rows (source.frame.quantile_no_moments); "
        f"{_CATALOG}['from_moments']['quantile']",
        "declaration",
    ),
    ("from_moments", "REF:sequential.source.invalid"): Spec(
        "source_limited",
        "a cube replays an exported checkpoint; it holds no per-unit source to start a "
        "sequential process from",
        f"{_LIMITATIONS}: checkpoint replay",
        "declaration",
    ),
}


# Cell regions that refuse one hazard: the same hazard must raise one code on every route.
_HAZARDS: dict[str, str] = {
    "REF:frame.metric.cuped_does_apply": "quantile-cuped",
    "REF:arm.metric.quantile_cuped": "quantile-cuped",
    "REF:artifact.extension.invalid": "quantile-cuped",
    "REF:arm.metric.quantile_cluster": "quantile-cluster",
    "REF:source.frame.cluster_capability": "quantile-cluster",
    "REF:readout.metric.quantile_grain": "quantile-day-axis",
    "REF:breakout.quantile": "quantile-day-axis",
    "REF:query.builders.asof_group_summary_metric_type_not_implemented": "quantile-day-axis",
    "REF:frame.asof.quantile_unsupported": "quantile-day-axis",
    "REF:sequential.route.unsupported#unbounded": "sequential-unbounded",
    "REF:sequential.route.unsupported#panel_unbounded": "sequential-unbounded",
    "REF:sequential.source.invalid": "sequential-unbounded",
    "REF:sequential.route.unsupported#metric_type": "sequential-quantile",
    "REF:sequential.route.unsupported#drop": "sequential-drop",
    "REF:sequential.route.unsupported#breakout": "sequential-breakout",
    "REF:frame.metric.winsorization_applies_type": "winsorization-non-mean",
    "REF:frame.metric.missing_impute": "impute-outcome",
    "REF:frame.validation.from_unit_panel": "panel-cuped-windowed",
    "REF:estimation.adjust_common.supported_ratio_metric": "observational-ratio",
    "REF:facade.analysis.observational_day_axis": "observational-day-axis",
    "REF:readout.view.observational": "observational-breakout",
    "REF:readout.margin.breakout": "margin-breakout",
    "REF:readout.metric.quantile_breakout": "quantile-breakout",
    "REF:readout.observational.quantile": "quantile-observational",
    "REF:readout.observational.quantile#asof_source": "quantile-day-axis",
    "REF:facade.analysis.clustered_day_axis": "cluster-day-axis",
    "REF:definition.invalid#cluster": "cluster-breakout",
    "REF:breakout.metric.daily_winsorization": "winsorization-day-axis",
    "REF:readout.inference.disjoint_slices": "sequential-day-slices",
    "REF:readout.metric.percentile_winsorization": "percentile-breakout",
}
# A hazard that several routes refuse with different codes is unfinished until they agree.
# The tracker named here owns the reconciliation; a hazard absent here may not diverge.
_HAZARD_TRACKERS: dict[str, str] = {
    "quantile-cuped": _T_ARTIFACT_QUANTILE_CUPED,
    "quantile-cluster": _T_ARTIFACT_QUANTILE_CUPED,
    "quantile-day-axis": _T_QUANTILE_DAY_AXIS,
    "sequential-unbounded": _T_SEQUENTIAL_UNBOUNDED,
}


def verdict(ingress: str, why: str) -> Verdict:
    override = _OVERRIDES.get((ingress, why))
    spec = override or _SPECS[why]
    return Verdict(
        _outcome(why, spec),
        spec.status,
        spec.reason,
        spec.authority,
        spec.stage,
        spec.tracker,
        hazard=None if override else _HAZARDS.get(why),
    )


_CATALOG_CAPABILITY = {
    "cuped": "cuped",
    "cluster": "cluster",
    "sequential": "sequential",
    "observational": "observational",
}
_OPTION_AUTHORITY = {
    "none": "",
    "cuped": f"{_LIMITATIONS}: Mean/Ratio CUPED rows",
    "winsor_fixed": f"{_LIMITATIONS}: fixed-threshold winsorization row; docs/guides/metric-types.md",
    "winsor_percentile": "docs/guides/metric-types.md (percentile winsorization on a mean)",
    "cluster": f"{_LIMITATIONS}: clustered rows",
    "sequential": f"{_LIMITATIONS}: sequential rows",
    "observational": f"{_LIMITATIONS}: Observational IPTW row",
    "ni_margin": "increment/semantics/models.py::ExperimentMetric.margin (guardrail margin)",
}
_OPTION_PREREQUISITE = {
    "none": "",
    "cuped": "a pre-period covariate (n_pre_periods on the warehouse, covariate= on a frame)",
    "winsor_fixed": "a mean metric with a declared upper bound",
    "winsor_percentile": "a mean metric with a declared upper percentile and positive outcomes",
    "cluster": "a declared cluster column with at least 40 clusters",
    "sequential": "a registered asymptotic or exact Bernoulli inference, a declared allocation "
    "and windows covered by the common reveal window",
    "observational": "a declared pre-exposure covariate",
    "ni_margin": "a declared relative margin on a guardrail",
}
_INGRESS_AUTHORITY = {
    "from_definitions": "Analysis.from_definitions",
    "from_unit_day_artifact": "Analysis.from_unit_day_artifact docstring",
    "from_unit_summary": "Analysis.from_unit_summary docstring",
    "from_unit_panel": "Analysis.from_unit_panel docstring",
    "from_moments": "Analysis.from_moments docstring",
    "from_switchback_panel": "Analysis.from_switchback_panel docstring",
}
# Catalog cells that read `refused` for a request the matrix runs, with the reason the
# catalog's probe differs. Any other refused catalog cell contradicts a supported verdict.
_CATALOG_SUPERSEDED: dict[tuple[str, str], str] = {
    ("sequential", "mean"): "the catalog probes a Gaussian registration; the matrix registers "
    "asymptotic_mean over bounded windows (the catalog's own advisory names this route)",
    ("sequential", "ratio"): "the catalog probes the exact Gaussian ratio likelihood; the matrix "
    "registers asymptotic_mean (the catalog's own advisory names this route)",
    ("observational", "retention"): "the catalog probes the frame summary seam; the warehouse "
    "routes read retention through the native unit frame",
    ("cluster", "quantile"): "the catalog probes the frame; see the quantile-cluster hazard",
}


def catalog_capabilities(cell: Cell, method: str) -> tuple[str, ...]:
    """The `tests/compatibility_catalog.py::MATRIX` capabilities a leg exercises.

    A day-axis value series is an absolute per-day mean: no adjustment option applies to it.
    """
    caps = [{"run": "estimate", "breakout": "breakout"}.get(cell.view, "daily_asof")]
    if method != "values" and cell.option in _CATALOG_CAPABILITY:
        caps.append(_CATALOG_CAPABILITY[cell.option])
    return tuple(caps)


def supported_verdict(cell: Cell, ingress: str, method: str) -> Verdict:
    """Why a running region is supported: the capability claims it rests on, and what the
    request must declare for the ingress to run it."""
    caps = catalog_capabilities(cell, method)
    claims = "; ".join(f"{_CATALOG}[{cap!r}][{cell.base!r}]" for cap in caps)
    authorities = [
        claims,
        "" if method == "values" else _OPTION_AUTHORITY[cell.option],
        _INGRESS_AUTHORITY[ingress],
    ]
    if cell.view in ("daily", "asof"):
        doc = _RUN_ASOF_LIFT if cell.view == "asof" else _RUN_DAILY_LIFT
        authorities.append(doc if method == "lift" else f"Analysis.run_{cell.view} docstring")
    needs = [
        "a bounded exposure window and day boundary" if cell.windowed else "",
        "an exposure date and an observation band" if cell.base == "retention" else "",
        "a declared breakout dimension" if cell.view == "breakout" else "",
        "" if method == "values" else _OPTION_PREREQUISITE[cell.option],
    ]
    prerequisites = "; ".join(n for n in needs if n)
    reason = (
        "runs; its fixed-horizon contrast is a different estimand, exercised on its own and "
        "never row-compared with the matched-arm ingresses"
        if ingress == "from_switchback_panel"
        else "runs and agrees with every other running matched-arm ingress"
    )
    if prerequisites:
        reason += f" once the request declares {prerequisites}"
    superseded = [
        note for cap in caps if (note := _CATALOG_SUPERSEDED.get((cap, cell.base))) is not None
    ]
    if superseded:
        reason += f" (catalog reads refused: {' / '.join(superseded)})"
    return Verdict(
        Runs(),
        "supported",
        reason,
        "; ".join(a for a in authorities if a),
    )


# fmt: off
RULES: dict[str, tuple[Box, ...]] = {
    "from_definitions": (
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:arm.metric.quantile_cluster", "quantile windowed_quantile", "run", "cluster", "error zero", "rows"),
        box("REF:arm.metric.quantile_cuped", "quantile windowed_quantile", "run", "cuped", "error zero", "rows"),
        box("REF:definition.active.metric_window_days", "windowed_active", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:definition.invalid#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "cluster", "error zero", "rows"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:definition.total.metric_window_days", "windowed_total", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero", "rows"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero", "rows"),
        box("REF:readout.metric.quantile_alternative", "quantile windowed_quantile", "run", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.quantile_breakout", "quantile windowed_quantile", "breakout", "none cuped ni_margin", "error zero", "rows"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "observational", "error zero", "rows"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "run breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "run breakout", "sequential", "error zero", "rows"),
        box("REF:readout.observational.quantile", "quantile windowed_quantile", "run", "observational", "error zero", "rows"),
        box("EXC:ValidationError#missing", "*", "run breakout", "*", "drop impute", "rows"),
        box("REF:definition.invalid#report_only", "total active", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "run breakout", "*", "error zero", "rows"),
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "values"),
        box("REF:breakout.quantile", "quantile windowed_quantile", "daily asof", "none cuped observational ni_margin", "error zero", "values"),
        box("REF:definition.active.metric_window_days", "windowed_active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.invalid#report_only", "total active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.total.metric_window_days", "windowed_total", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero", "values"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "daily asof", "sequential", "error zero", "values"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "values"),
        box("EXC:ValidationError#missing", "*", "daily asof", "*", "drop impute", "values"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped sequential observational ni_margin", "error zero", "values"),
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "lift"),
        box("REF:breakout.quantile", "quantile windowed_quantile", "daily asof", "none cuped observational ni_margin", "error zero", "lift"),
        box("REF:definition.active.metric_window_days", "windowed_active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.invalid#report_only", "total active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.total.metric_window_days", "windowed_total", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero", "lift"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "observational", "error zero", "lift"),
        box("REF:readout.inference.disjoint_slices", "retention windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero", "lift"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "daily asof", "sequential", "error zero", "lift"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "lift"),
        box("EXC:ValidationError#missing", "*", "daily asof", "*", "drop impute", "lift"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped sequential ni_margin", "error zero", "lift"),
    ),
    "from_unit_day_artifact": (
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:arm.metric.quantile_cluster", "quantile windowed_quantile", "run", "cluster", "error zero", "rows"),
        box("REF:artifact.extension.invalid", "quantile windowed_quantile", "run breakout", "cuped", "error zero", "rows"),
        box("REF:definition.active.metric_window_days", "windowed_active", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:definition.invalid#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "cluster", "error zero", "rows"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:definition.total.metric_window_days", "windowed_total", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero", "rows"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero", "rows"),
        box("REF:readout.metric.quantile_alternative", "quantile windowed_quantile", "run", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.quantile_breakout", "quantile", "breakout", "none ni_margin", "error zero", "rows"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "breakout", "observational", "error zero", "rows"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "run breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "run breakout", "sequential", "error zero", "rows"),
        box("EXC:ValidationError#missing", "*", "run breakout", "*", "drop impute", "rows"),
        box("REF:definition.invalid#report_only", "total active", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout", "none ni_margin", "error zero", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "breakout", "observational", "error zero", "rows"),
        box("REF:readout.observational.quantile", "quantile windowed_quantile", "run", "observational", "error zero", "rows"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "error zero", "rows"),
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:artifact.extension.invalid", "quantile windowed_quantile", "daily asof", "cuped", "error zero", "values"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "values"),
        box("REF:definition.active.metric_window_days", "windowed_active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.invalid#report_only", "total active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:definition.total.metric_window_days", "windowed_total", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "values"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero", "values"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "asof", "none observational ni_margin", "error zero", "values"),
        box("REF:query.builders.asof_group_summary_metric_type_not_implemented", "quantile", "asof", "none ni_margin", "error zero", "values"),
        box("REF:readout.observational.quantile#asof_source", "quantile", "asof", "observational", "error zero", "values"),
        box("REF:readout.metric.quantile_grain", "quantile windowed_quantile", "daily", "none observational ni_margin", "error zero", "values"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "daily asof", "sequential", "error zero", "values"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "values"),
        box("EXC:ValidationError#missing", "*", "daily asof", "*", "drop impute", "values"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped sequential observational ni_margin", "error zero", "values"),
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:artifact.extension.invalid", "quantile windowed_quantile", "daily asof", "cuped", "error zero", "lift"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "lift"),
        box("REF:definition.active.metric_window_days", "windowed_active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.invalid#report_only", "total active", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:definition.total.metric_window_days", "windowed_total", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero", "lift"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero", "lift"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "observational", "error zero", "lift"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "asof", "none ni_margin", "error zero", "lift"),
        box("REF:query.builders.asof_group_summary_metric_type_not_implemented", "quantile", "asof", "none ni_margin", "error zero", "lift"),
        box("REF:readout.inference.disjoint_slices", "retention windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero", "lift"),
        box("REF:readout.metric.quantile_grain", "quantile windowed_quantile", "daily", "none ni_margin", "error zero", "lift"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "daily asof", "sequential", "error zero", "lift"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "lift"),
        box("EXC:ValidationError#missing", "*", "daily asof", "*", "drop impute", "lift"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped sequential ni_margin", "error zero", "lift"),
    ),
    "from_unit_summary": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "*", "rows"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio", "run", "observational", "error zero drop", "rows"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "run breakout", "cuped", "*", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout", "none cluster sequential observational ni_margin", "*", "rows"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "run breakout", "*", "*", "rows"),
        box("REF:readout.metric.quantile_alternative", "quantile", "run", "ni_margin", "error zero drop", "rows"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio", "run breakout", "sequential", "drop", "rows"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "run breakout", "sequential", "error zero drop", "rows"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "run breakout", "sequential", "error zero", "rows"),
        box("REF:source.frame.cluster_capability", "quantile", "run breakout", "cluster", "error zero drop", "rows"),
        box("REF:source.frame.constructor#retention", "retention", "run breakout", "none cuped cluster sequential observational ni_margin", "error zero drop", "rows"),
        box("REF:readout.observational.quantile", "quantile", "run", "observational", "error zero drop", "rows"),
        box("REF:facade.analysis.operation", "mean conversion ratio quantile", "breakout", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop", "rows"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "impute", "rows"),
        box("REF:source.frame.constructor#window", "windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "error zero drop", "rows"),
        box("RUNS", "mean conversion ratio quantile", "run", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop", "rows"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "values"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "*", "*", "values"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio", "daily asof", "sequential", "drop", "values"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "values"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "daily asof", "sequential", "error zero", "values"),
        box("REF:source.frame.cluster_capability", "quantile", "daily asof", "cluster", "error zero drop", "values"),
        box("REF:source.frame.constructor#retention", "retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero drop", "values"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio quantile", "daily asof", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop", "values"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "values"),
        box("REF:source.frame.constructor#window", "windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "error zero drop", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "lift"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "*", "*", "lift"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio", "daily asof", "sequential", "drop", "lift"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "lift"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "daily asof", "sequential", "error zero", "lift"),
        box("REF:source.frame.cluster_capability", "quantile", "daily asof", "cluster", "error zero drop", "lift"),
        box("REF:source.frame.constructor#retention", "retention", "daily asof", "none cuped cluster sequential observational ni_margin", "error zero drop", "lift"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio quantile", "daily asof", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop", "lift"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "lift"),
        box("REF:source.frame.constructor#window", "windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "error zero drop", "lift"),
    ),
    "from_unit_panel": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "*", "rows"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero", "rows"),
        box("REF:estimation.winsor.raw_state_required", "mean windowed_mean", "run", "winsor_percentile", "error zero", "rows"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "run breakout", "cuped", "*", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout", "none cluster sequential observational ni_margin", "*", "rows"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "run breakout", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "*", "rows"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "run breakout", "cuped", "error zero drop", "rows"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero", "rows"),
        box("REF:readout.metric.quantile_alternative", "quantile", "run", "ni_margin", "error zero", "rows"),
        box("REF:readout.metric.quantile_breakout", "quantile", "breakout", "none ni_margin", "error zero", "rows"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "breakout", "observational", "error zero", "rows"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "run breakout", "sequential", "drop", "rows"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "run breakout", "sequential", "error zero drop", "rows"),
        box("REF:sequential.route.unsupported#panel_unbounded", "mean conversion ratio", "run breakout", "sequential", "error zero", "rows"),
        box("REF:source.frame.unit_frame_panel", "retention windowed_mean windowed_conversion", "run", "observational", "error zero", "rows"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "impute", "rows"),
        box("REF:frame.missing_policy.panel_drop", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "none cuped winsor_fixed winsor_percentile observational ni_margin", "drop", "rows"),
        box("REF:readout.observational.quantile", "quantile", "run", "observational", "error zero", "rows"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "none cuped winsor_fixed sequential observational ni_margin", "error zero", "rows"),
        box("REF:source.frame_panel.cluster_grain", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "run breakout", "cluster", "*", "rows"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.asof.quantile_unsupported", "quantile", "asof", "none observational ni_margin", "error zero", "values"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "values"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "*", "values"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "cuped", "error zero drop", "values"),
        box("REF:readout.metric.quantile_grain", "quantile", "daily", "none observational ni_margin", "error zero", "values"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "sequential", "drop", "values"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "values"),
        box("REF:sequential.route.unsupported#panel_unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "values"),
        box("REF:source.frame.retention_daily", "retention", "daily", "none sequential observational ni_margin", "error zero", "values"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "values"),
        box("REF:frame.missing_policy.panel_drop", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped winsor_fixed winsor_percentile observational ni_margin", "drop", "values"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped sequential observational ni_margin", "error zero", "values"),
        box("REF:source.frame_panel.cluster_grain", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "values"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero", "lift"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "lift"),
        box("REF:estimation.cuped.arm_no_covariate", "mean conversion ratio", "daily asof", "cuped", "error zero", "lift"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "observational", "error zero", "lift"),
        box("REF:frame.asof.quantile_unsupported", "quantile", "asof", "none ni_margin", "error zero", "lift"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "lift"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "*", "lift"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "cuped", "error zero drop", "lift"),
        box("REF:readout.inference.disjoint_slices", "retention windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero", "lift"),
        box("REF:readout.metric.quantile_grain", "quantile", "daily", "none ni_margin", "error zero", "lift"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "sequential", "drop", "lift"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "lift"),
        box("REF:sequential.route.unsupported#panel_unbounded", "mean conversion ratio", "daily asof", "sequential", "error zero", "lift"),
        box("REF:source.frame.retention_daily", "retention", "daily", "none ni_margin", "error zero", "lift"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "lift"),
        box("REF:frame.missing_policy.panel_drop", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped winsor_fixed winsor_percentile observational ni_margin", "drop", "lift"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none sequential ni_margin", "error zero", "lift"),
        box("REF:source.frame_panel.cluster_grain", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "lift"),
    ),
    "from_moments": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "*", "rows"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "run breakout", "cuped", "*", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout", "none cluster sequential observational ni_margin", "*", "rows"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "run breakout", "cuped", "drop", "rows"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero", "rows"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "run breakout", "sequential", "drop", "rows"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "run breakout", "sequential", "error zero drop", "rows"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "run breakout", "sequential", "error zero", "rows"),
        box("REF:source.frame.cluster_capability", "quantile", "run breakout", "cluster", "error zero drop", "rows"),
        box("REF:readout.observational.quantile", "quantile", "run", "observational", "error zero drop", "rows"),
        box("REF:facade.analysis.operation#observational_quantile", "quantile", "breakout", "observational", "error zero drop", "rows"),
        box("REF:source.frame.quantile_no_moments", "quantile", "run breakout", "none ni_margin", "error zero drop", "rows"),
        box("REF:source.moments.cluster_grain", "mean conversion ratio", "run breakout", "cluster", "error zero drop", "rows"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "impute", "rows"),
        box("REF:frame.missing_policy.panel_drop", "retention windowed_mean windowed_conversion windowed_ratio", "run breakout", "none winsor_fixed winsor_percentile observational ni_margin", "drop", "rows"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero drop", "rows"),
        box("REF:estimation.winsor.raw_state_required", "mean windowed_mean", "run", "winsor_percentile", "error zero drop", "rows"),
        box("REF:facade.analysis.operation", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "none cuped winsor_fixed winsor_percentile observational ni_margin", "error zero drop", "rows"),
        box("REF:source.frame_panel.cluster_grain", "retention windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "run breakout", "cluster", "*", "rows"),
        box("REF:source.moments.covariate_unavailable", "mean conversion retention windowed_mean windowed_conversion", "run", "observational", "error zero drop", "rows"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "run", "none cuped winsor_fixed sequential ni_margin", "error zero drop", "rows"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "run breakout", "*", "*", "rows"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "values"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "cuped", "drop", "values"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "sequential", "drop", "values"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "values"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "daily asof", "sequential", "error zero", "values"),
        box("REF:source.frame.cluster_capability", "quantile", "daily asof", "cluster", "error zero drop", "values"),
        box("REF:facade.analysis.no_definitions#observational_quantile", "quantile", "daily asof", "observational", "error zero drop", "values"),
        box("REF:source.frame.quantile_no_moments", "quantile", "daily asof", "none ni_margin", "error zero drop", "values"),
        box("REF:source.moments.cluster_grain", "mean conversion ratio", "daily asof", "cluster", "error zero drop", "values"),
        box("REF:source.moments.grain", "retention windowed_mean windowed_conversion windowed_ratio", "asof", "sequential", "error zero", "values"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "values"),
        box("REF:frame.missing_policy.panel_drop", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none winsor_fixed winsor_percentile observational ni_margin", "drop", "values"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "error zero drop", "values"),
        box("REF:source.frame_panel.cluster_grain", "retention windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "values"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "*", "*", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "lift"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "cuped", "drop", "lift"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "sequential", "drop", "lift"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "daily asof", "sequential", "error zero drop", "lift"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "daily asof", "sequential", "error zero", "lift"),
        box("REF:source.frame.cluster_capability", "quantile", "daily asof", "cluster", "error zero drop", "lift"),
        box("REF:facade.analysis.no_definitions#observational_quantile", "quantile", "daily asof", "observational", "error zero drop", "lift"),
        box("REF:source.frame.quantile_no_moments", "quantile", "daily asof", "none ni_margin", "error zero drop", "lift"),
        box("REF:source.moments.cluster_grain", "mean conversion ratio", "daily asof", "cluster", "error zero drop", "lift"),
        box("RUNS", "retention windowed_mean windowed_conversion windowed_ratio", "asof", "sequential", "error zero", "lift"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "lift"),
        box("REF:frame.missing_policy.panel_drop", "retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none winsor_fixed winsor_percentile observational ni_margin", "drop", "lift"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "error zero drop", "lift"),
        box("REF:source.frame_panel.cluster_grain", "retention windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "lift"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "*", "*", "lift"),
    ),
    "from_switchback_panel": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "run breakout", "none cuped cluster sequential observational ni_margin", "*", "rows"),
        box("REF:facade.analysis.contrast_unavailable", "mean conversion", "breakout", "none", "error", "rows"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "run breakout", "cuped", "*", "rows"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout", "none cluster sequential observational ni_margin", "*", "rows"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "run breakout", "winsor_fixed winsor_percentile", "*", "rows"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "run breakout", "none cuped winsor_fixed winsor_percentile sequential ni_margin", "*", "rows"),
        box("REF:source.frame.switchback.metric#method", "mean conversion windowed_mean windowed_conversion", "run breakout", "cuped", "error", "rows"),
        box("REF:source.frame.switchback.metric#missing", "mean conversion windowed_mean windowed_conversion", "run breakout", "none cuped sequential ni_margin", "zero drop", "rows"),
        box("REF:source.frame.switchback.metric#window", "windowed_mean windowed_conversion", "run breakout", "none sequential ni_margin", "error", "rows"),
        box("REF:source.frame.switchback.metric#winsor", "mean windowed_mean", "run breakout", "winsor_fixed winsor_percentile", "error zero drop", "rows"),
        box("REF:source.frame.switchback.plan#inference", "mean conversion", "run breakout", "sequential", "error", "rows"),
        box("REF:source.frame.switchback.plan#margin", "mean conversion", "run breakout", "ni_margin", "error", "rows"),
        box("RUNS", "mean conversion", "run", "none", "error", "rows"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "run breakout", "*", "impute", "rows"),
        box("REF:source.frame.switchback.metric#type", "ratio retention quantile windowed_ratio", "run breakout", "none cuped sequential ni_margin", "error zero drop", "rows"),
        box("EXC:TypeError#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "run breakout", "cluster", "*", "rows"),
        box("REF:source.frame.switchback.identification", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "run breakout", "observational", "*", "rows"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "values"),
        box("REF:facade.analysis.contrast_unavailable", "mean conversion", "daily asof", "none", "error", "values"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "values"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "values"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "values"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "none cuped winsor_fixed winsor_percentile sequential ni_margin", "*", "values"),
        box("REF:source.frame.switchback.metric#method", "mean conversion windowed_mean windowed_conversion", "daily asof", "cuped", "error", "values"),
        box("REF:source.frame.switchback.metric#missing", "mean conversion windowed_mean windowed_conversion", "daily asof", "none cuped sequential ni_margin", "zero drop", "values"),
        box("REF:source.frame.switchback.metric#window", "windowed_mean windowed_conversion", "daily asof", "none sequential ni_margin", "error", "values"),
        box("REF:source.frame.switchback.metric#winsor", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero drop", "values"),
        box("REF:source.frame.switchback.plan#inference", "mean conversion", "daily asof", "sequential", "error", "values"),
        box("REF:source.frame.switchback.plan#margin", "mean conversion", "daily asof", "ni_margin", "error", "values"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "values"),
        box("REF:source.frame.switchback.metric#type", "ratio retention quantile windowed_ratio", "daily asof", "none cuped sequential ni_margin", "error zero drop", "values"),
        box("EXC:TypeError#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "values"),
        box("REF:source.frame.switchback.identification", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "observational", "*", "values"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "daily asof", "none cuped cluster sequential observational ni_margin", "*", "lift"),
        box("REF:facade.analysis.contrast_unavailable", "mean conversion", "daily asof", "none", "error", "lift"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "daily asof", "cuped", "*", "lift"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "daily asof", "none cluster sequential observational ni_margin", "*", "lift"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "daily asof", "winsor_fixed winsor_percentile", "*", "lift"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "daily asof", "none cuped winsor_fixed winsor_percentile sequential ni_margin", "*", "lift"),
        box("REF:source.frame.switchback.metric#method", "mean conversion windowed_mean windowed_conversion", "daily asof", "cuped", "error", "lift"),
        box("REF:source.frame.switchback.metric#missing", "mean conversion windowed_mean windowed_conversion", "daily asof", "none cuped sequential ni_margin", "zero drop", "lift"),
        box("REF:source.frame.switchback.metric#window", "windowed_mean windowed_conversion", "daily asof", "none sequential ni_margin", "error", "lift"),
        box("REF:source.frame.switchback.metric#winsor", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero drop", "lift"),
        box("REF:source.frame.switchback.plan#inference", "mean conversion", "daily asof", "sequential", "error", "lift"),
        box("REF:source.frame.switchback.plan#margin", "mean conversion", "daily asof", "ni_margin", "error", "lift"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "daily asof", "*", "impute", "lift"),
        box("REF:source.frame.switchback.metric#type", "ratio retention quantile windowed_ratio", "daily asof", "none cuped sequential ni_margin", "error zero drop", "lift"),
        box("EXC:TypeError#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "cluster", "*", "lift"),
        box("REF:source.frame.switchback.identification", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "daily asof", "observational", "*", "lift"),
    ),
}
# fmt: on


def classify(cell: Cell) -> Disposition:
    """The disposition of *cell*; ``LookupError`` for a value outside the axes."""
    if (
        cell.metric not in METRICS
        or cell.view not in VIEWS
        or cell.option not in OPTIONS
        or cell.day_boundary not in DAY_BOUNDARIES
        or cell.missing not in MISSING
    ):
        raise LookupError(f"{cell.id} lies outside the enumerated axes")
    legs: dict[str, dict[str, Verdict]] = {}
    for method in methods(cell):
        verdicts = {}
        for ingress in INGRESSES:
            rule = next((b for b in RULES[ingress] if b.matches(cell, method)), None)
            if rule is None:
                raise LookupError(f"no {ingress} rule classifies {cell.id} ({method})")
            verdicts[ingress] = (
                supported_verdict(cell, ingress, method)
                if rule.why == "RUNS"
                else verdict(ingress, rule.why)
            )
        legs[method] = _reconcile_hazards(verdicts)
    return Disposition(legs)


def _hazard_codes(verdicts: Mapping[str, Verdict]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for v in verdicts.values():
        if v.hazard and isinstance(v.outcome, Refuses):
            groups.setdefault(v.hazard, set()).add(v.outcome.code)
    return groups


def _reconcile_hazards(verdicts: dict[str, Verdict]) -> dict[str, Verdict]:
    """One hazard refused with several codes is unfinished on every route that raises it.

    The divergence is the defect, not any single route's code, so each route of the hazard
    becomes ``unfinished`` under the hazard's tracker. A diverging hazard with no tracker
    is a fault of the matrix (``LookupError``), never silently classified.
    """
    out = dict(verdicts)
    for hazard, codes in _hazard_codes(verdicts).items():
        if len(codes) < 2:
            continue
        tracker = _HAZARD_TRACKERS.get(hazard)
        if tracker is None:
            raise LookupError(f"hazard {hazard!r} raises several codes {sorted(codes)}: no tracker")
        for name, v in verdicts.items():
            if v.hazard == hazard and v.status == "construction_limited":
                out[name] = replace(
                    v,
                    status="unfinished",
                    tracker=tracker,
                    reason=f"{v.reason}; the hazard raises {len(codes)} codes across routes "
                    f"({', '.join(sorted(codes))}), one hazard needs one code",
                )
    return out


def check_disposition(cell: Cell, disposition: Disposition) -> None:
    """Raise ``AssertionError`` when a disposition breaks the matrix's own contract.

    Per leg: a refuser beside a runner needs an explanatory status; one hazard refused with
    several codes, across stages and ingresses alike, is unfinished everywhere it is raised;
    and a supported verdict cannot contradict a capability the compatibility catalog marks
    not applicable or refused without a recorded supersession.
    """
    for method, verdicts in disposition.legs.items():
        runners = [n for n, v in verdicts.items() if isinstance(v.outcome, Runs)]
        for name, v in verdicts.items():
            if runners and not isinstance(v.outcome, Runs):
                assert v.status in _EXPLAINS_SPLIT, (
                    f"{cell.id}/{method}: {name} {v.status} cannot explain a split"
                )
        for hazard, codes in _hazard_codes(verdicts).items():
            if len(codes) > 1:
                for name, v in verdicts.items():
                    if v.hazard == hazard:
                        assert v.status == "unfinished" and v.tracker, (
                            f"{cell.id}/{method}: hazard {hazard!r} raises {sorted(codes)} "
                            f"but {name} is {v.status} without a tracker"
                        )
        if runners:
            _check_supported_against_catalog(cell, method)


def _check_supported_against_catalog(cell: Cell, method: str) -> None:
    from tests.compatibility_catalog import MATRIX

    for cap in catalog_capabilities(cell, method):
        status = MATRIX[cap][cell.base].status
        assert status != "na", f"{cell.id}: runs where the catalog says {cap} is not applicable"
        assert status != "refused" or (cap, cell.base) in _CATALOG_SUPERSEDED, (
            f"{cell.id}: runs where the catalog refuses {cap} x {cell.base} (no recorded reason)"
        )


_EXPLAINS_SPLIT = frozenset(
    {"source_limited", "not_expressible", "construction_limited", "unfinished", "unsound"}
)
