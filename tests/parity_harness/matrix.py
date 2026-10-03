"""The enumerated parity matrix: axes, cells, and the disposition of every cell.

``tests/test_parity_matrix.py`` runs every cell of the cross product

    metric x view x option x day_boundary x missing

through all six ingress constructors and checks the result against the
``Disposition`` that ``classify`` returns for it. This module is the single
machine-readable source of those dispositions; nothing else restates them, and
``tests/parity_harness/COVERAGE.md`` only counts them.

A cell's disposition is one ``Verdict`` per ingress: what it does (``Runs``,
``Refuses(code)`` or ``StructuralAbsence``), the status that explains it, a reason
and an authority independent of the refusal itself. The statuses are

``supported``            runs, and agrees with every other running ingress within the
                         runner's 1e-9 relative tolerance
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
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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
    """The ingress produces rows; they are compared with every other running ingress."""


@dataclass(frozen=True)
class Refuses:
    """The ingress raises ``CodedError`` with exactly this code."""

    code: str


@dataclass(frozen=True)
class StructuralAbsence:
    """The constructor or schema cannot express the request.

    The runner attempts the declaration with the unsupported field or keyword and expects
    exactly ``error`` (``TypeError`` for a keyword, pydantic's ``ValidationError`` for a
    schema field); no refusal code exists to record.
    """

    fact: str
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

    def __post_init__(self) -> None:
        if not self.reason or not self.authority:
            raise ValueError("a verdict needs a reason and an authority")
        if self.status == "unfinished" and not self.tracker:
            raise ValueError("an unfinished verdict needs a tracker")
        if self.status == "unsound" and not self.derivation:
            raise ValueError("an unsound verdict needs a cited derivation")
        if isinstance(self.outcome, Runs) != (self.status == "supported"):
            raise ValueError("only a running ingress is supported")


@dataclass(frozen=True)
class Disposition:
    verdicts: Mapping[str, Verdict]

    def __post_init__(self) -> None:
        if set(self.verdicts) != set(INGRESSES):
            raise ValueError(f"verdicts must cover exactly the six ingresses: {set(self.verdicts)}")

    @property
    def outcomes(self) -> dict[str, Outcome]:
        return {name: verdict.outcome for name, verdict in self.verdicts.items()}

    @property
    def runners(self) -> tuple[str, ...]:
        return tuple(n for n, v in self.verdicts.items() if isinstance(v.outcome, Runs))

    @property
    def status(self) -> Status:
        """The cell's status: the most significant verdict.

        ``unfinished`` or ``unsound`` anywhere wins; else ``supported`` when anything
        runs; else ``not_expressible`` when every refusal is a declaration the axis value
        cannot make; else ``construction_limited`` when any refusal is, else
        ``source_limited``.
        """
        statuses = {v.status for v in self.verdicts.values()}
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
        return tuple(sorted({v.tracker for v in self.verdicts.values() if v.tracker}))


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

    def matches(self, cell: Cell) -> bool:
        return (
            (self.metric is None or cell.metric in self.metric)
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
) -> Box:
    """A ``Box`` from space-separated axis values; ``*`` matches every value."""

    def axis(text: str) -> tuple[str, ...] | None:
        return None if text == "*" else tuple(text.split())

    return Box(why, axis(metric), axis(view), axis(option), axis(missing))


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

# Trackers name issues to be opened from the t8tc report; each carries the code now raised.
_T_QUANTILE_DAY_AXIS = "kata:NEW/quantile-day-axis-refusal-codes"
_T_WINDOWED_QUANTILE = "kata:NEW/windowed-quantile-frame-and-artifact"
_T_ARTIFACT_QUANTILE_CUPED = "kata:NEW/artifact-quantile-cuped-refusal-code"
_T_SEQUENTIAL_UNBOUNDED = "kata:NEW/sequential-unbounded-window-refusal-code"
_T_QUANTILE_OBSERVATIONAL = "kata:NEW/quantile-observational-route-split"
_T_QUANTILE_ONE_SIDED = "kata:NEW/quantile-one-sided-alternative"

_SPECS: dict[str, Spec] = {
    "RUNS": Spec(
        "supported",
        "runs and agrees with every other running ingress",
        _RUNNER,
    ),
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
    "EXC:TypeError": Spec(
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
    "REF:source.native.operation": Spec(
        "unfinished",
        "an adjusted quantile reads per-unit rows on the panel and artifact routes, but the "
        "native source refuses the moments operation instead",
        "increment/query/native_source.py (quantile is served through unit_frame)",
        tracker=_T_QUANTILE_OBSERVATIONAL,
    ),
    "REF:source.frame.quantile_no_moments": Spec(
        "unfinished",
        "a quantile has no moments representation, yet the panel and artifact routes serve "
        "an adjusted quantile through a unit frame",
        "increment/frame.py (quantile is served through unit_frame); "
        f"{_CATALOG}['from_moments']['quantile']",
        tracker=_T_QUANTILE_OBSERVATIONAL,
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
    name = rest.partition("#")[0]
    if kind == "REF":
        return Refuses(name)
    return StructuralAbsence(
        spec.fact, {"ValidationError": ValidationError, "TypeError": TypeError}[name]
    )


# The same observed signature is a different thing on one ingress.
_OVERRIDES: dict[tuple[str, str], Spec] = {
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


def verdict(ingress: str, why: str) -> Verdict:
    spec = _OVERRIDES.get((ingress, why)) or _SPECS[why]
    return Verdict(
        _outcome(why, spec),
        spec.status,
        spec.reason,
        spec.authority,
        spec.stage,
        spec.tracker,
    )


# fmt: off
RULES: dict[str, tuple[Box, ...]] = {
    "from_definitions": (
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:arm.metric.quantile_cluster", "quantile windowed_quantile", "run", "cluster", "error zero"),
        box("REF:arm.metric.quantile_cuped", "quantile windowed_quantile", "run", "cuped", "error zero"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero"),
        box("REF:breakout.quantile", "quantile windowed_quantile", "daily asof", "none cuped observational ni_margin", "error zero"),
        box("REF:definition.active.metric_window_days", "windowed_active", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:definition.invalid#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "cluster", "error zero"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:definition.total.metric_window_days", "windowed_total", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "observational", "error zero"),
        box("REF:readout.inference.disjoint_slices", "retention windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero"),
        box("REF:readout.metric.quantile_alternative", "quantile windowed_quantile", "run", "ni_margin", "error zero"),
        box("REF:readout.metric.quantile_breakout", "quantile windowed_quantile", "breakout", "none cuped ni_margin", "error zero"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "observational", "error zero"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "*", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "*", "sequential", "error zero"),
        box("REF:source.native.operation", "quantile windowed_quantile", "run", "observational", "error zero"),
        box("EXC:ValidationError#missing", "*", "*", "*", "drop impute"),
        box("REF:definition.invalid#report_only", "total active", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "*", "*", "error zero"),
    ),
    "from_unit_day_artifact": (
        box("EXC:ValidationError#winsorization", "conversion ratio retention quantile total active windowed_conversion windowed_ratio windowed_retention windowed_quantile windowed_total windowed_active", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:arm.metric.quantile_cluster", "quantile windowed_quantile", "run", "cluster", "error zero"),
        box("REF:artifact.extension.invalid", "quantile windowed_quantile", "*", "cuped", "error zero"),
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero"),
        box("REF:definition.active.metric_window_days", "windowed_active", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:definition.invalid#cluster", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_quantile", "breakout", "cluster", "error zero"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:definition.total.metric_window_days", "windowed_total", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero"),
        box("REF:facade.analysis.clustered_day_axis", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio windowed_quantile", "daily asof", "cluster", "error zero"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "observational", "error zero"),
        box("REF:query.builders.asof_group_summary_metric_type_not_implemented", "quantile", "asof", "none observational ni_margin", "error zero"),
        box("REF:readout.inference.disjoint_slices", "retention windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero"),
        box("REF:readout.metric.quantile_alternative", "quantile windowed_quantile", "run", "ni_margin", "error zero"),
        box("REF:readout.metric.quantile_breakout", "quantile", "breakout", "none ni_margin", "error zero"),
        box("REF:readout.metric.quantile_grain", "quantile windowed_quantile", "daily", "none observational ni_margin", "error zero"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "breakout", "observational", "error zero"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#metric_type", "quantile windowed_quantile", "*", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#unbounded", "mean conversion ratio", "*", "sequential", "error zero"),
        box("EXC:ValidationError#missing", "*", "*", "*", "drop impute"),
        box("REF:definition.invalid#report_only", "total active", "*", "none cuped cluster sequential observational ni_margin", "error zero"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "run breakout asof", "none observational ni_margin", "error zero"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "*", "error zero"),
    ),
    "from_unit_summary": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "*"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio", "run", "observational", "error zero drop"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "*", "cuped", "*"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "*", "none cluster sequential observational ni_margin", "*"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "*", "*", "*"),
        box("REF:readout.metric.quantile_alternative", "quantile", "run", "ni_margin", "error zero drop"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio", "*", "sequential", "drop"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "*", "sequential", "error zero drop"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "*", "sequential", "error zero"),
        box("REF:source.frame.cluster_capability", "quantile", "*", "cluster", "error zero drop"),
        box("REF:source.frame.constructor#retention", "retention", "*", "none cuped cluster sequential observational ni_margin", "error zero drop"),
        box("REF:source.frame.quantile_no_moments", "quantile", "run", "observational", "error zero drop"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio quantile", "daily asof", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop"),
        box("REF:facade.analysis.operation", "mean conversion ratio quantile", "breakout", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "*", "impute"),
        box("REF:source.frame.constructor#window", "windowed_mean windowed_conversion windowed_ratio", "*", "*", "error zero drop"),
        box("RUNS", "mean conversion ratio quantile", "run", "none cuped winsor_fixed winsor_percentile cluster observational ni_margin", "error zero drop"),
    ),
    "from_unit_panel": (
        box("REF:breakout.metric.daily_winsorization", "mean windowed_mean", "daily asof", "winsor_fixed winsor_percentile", "error zero"),
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "*"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero"),
        box("REF:estimation.cuped.arm_no_covariate", "mean conversion ratio", "daily asof", "cuped", "error zero"),
        box("REF:estimation.winsor.raw_state_required", "mean windowed_mean", "run", "winsor_percentile", "error zero"),
        box("REF:frame.asof.quantile_unsupported", "quantile", "asof", "none observational ni_margin", "error zero"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "*", "cuped", "*"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "*", "none cluster sequential observational ni_margin", "*"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "*", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "*"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "*", "cuped", "error zero drop"),
        box("REF:readout.inference.disjoint_slices", "windowed_mean windowed_conversion windowed_ratio", "daily", "sequential", "error zero"),
        box("REF:readout.margin.breakout", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "ni_margin", "error zero"),
        box("REF:readout.metric.percentile_winsorization", "mean windowed_mean", "breakout", "winsor_percentile", "error zero"),
        box("REF:readout.metric.quantile_alternative", "quantile", "run", "ni_margin", "error zero"),
        box("REF:readout.metric.quantile_breakout", "quantile", "breakout", "none ni_margin", "error zero"),
        box("REF:readout.metric.quantile_grain", "quantile", "daily", "none observational ni_margin", "error zero"),
        box("REF:readout.view.observational", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "breakout", "observational", "error zero"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "*", "sequential", "drop"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "*", "sequential", "error zero drop"),
        box("REF:sequential.route.unsupported#panel_unbounded", "mean conversion ratio", "*", "sequential", "error zero"),
        box("REF:source.frame.retention_daily", "retention", "daily", "none sequential observational ni_margin", "error zero"),
        box("REF:source.frame.unit_frame_panel", "retention windowed_mean windowed_conversion", "run", "observational", "error zero"),
        box("REF:facade.analysis.observational_day_axis", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "observational", "error zero"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "*", "impute"),
        box("REF:frame.missing_policy.panel_drop", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "none cuped winsor_fixed winsor_percentile observational ni_margin", "drop"),
        box("REF:source.frame_panel.cluster_grain", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "*", "cluster", "*"),
        box("RUNS", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "none cuped winsor_fixed sequential observational ni_margin", "error zero"),
    ),
    "from_moments": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "*"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "*", "cuped", "*"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "*", "none cluster sequential observational ni_margin", "*"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:frame.validation.from_unit_panel", "retention windowed_mean windowed_conversion windowed_ratio", "*", "cuped", "error zero drop"),
        box("REF:sequential.route.unsupported#breakout", "retention windowed_mean windowed_conversion windowed_ratio", "breakout", "sequential", "error zero"),
        box("REF:sequential.route.unsupported#drop", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "*", "sequential", "drop"),
        box("REF:sequential.route.unsupported#metric_type", "quantile", "*", "sequential", "error zero drop"),
        box("REF:sequential.source.invalid", "mean conversion ratio", "*", "sequential", "error zero"),
        box("REF:source.frame.cluster_capability", "quantile", "*", "cluster", "error zero drop"),
        box("REF:source.frame.quantile_no_moments", "quantile", "*", "none observational ni_margin", "error zero drop"),
        box("REF:source.moments.cluster_grain", "mean conversion ratio", "*", "cluster", "error zero drop"),
        box("REF:source.moments.grain", "retention windowed_mean windowed_conversion windowed_ratio", "asof", "sequential", "error zero"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "*", "impute"),
        box("REF:frame.missing_policy.panel_drop", "retention windowed_mean windowed_conversion windowed_ratio", "*", "none winsor_fixed winsor_percentile observational ni_margin", "drop"),
        box("REF:estimation.adjust_common.supported_ratio_metric", "ratio windowed_ratio", "run", "observational", "error zero drop"),
        box("REF:estimation.winsor.raw_state_required", "mean windowed_mean", "run", "winsor_percentile", "error zero drop"),
        box("REF:facade.analysis.no_definitions", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "daily asof", "none cuped winsor_fixed winsor_percentile sequential observational ni_margin", "error zero drop"),
        box("REF:facade.analysis.operation", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "breakout", "none cuped winsor_fixed winsor_percentile observational ni_margin", "error zero drop"),
        box("REF:source.frame_panel.cluster_grain", "retention windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "*", "cluster", "*"),
        box("REF:source.moments.covariate_unavailable", "mean conversion retention windowed_mean windowed_conversion", "run", "observational", "error zero drop"),
        box("RUNS", "mean conversion ratio retention windowed_mean windowed_conversion windowed_ratio", "run", "none cuped winsor_fixed sequential ni_margin", "error zero drop"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "*", "*", "*"),
    ),
    "from_switchback_panel": (
        box("REF:definition.retention.metric_window_days", "windowed_retention", "*", "none cuped cluster sequential observational ni_margin", "*"),
        box("REF:facade.analysis.contrast_unavailable", "mean conversion", "breakout daily asof", "none", "error"),
        box("REF:frame.metric.cuped_does_apply", "quantile windowed_quantile", "*", "cuped", "*"),
        box("REF:frame.metric.window_days_supported", "windowed_quantile", "*", "none cluster sequential observational ni_margin", "*"),
        box("REF:frame.metric.winsorization_applies_type", "conversion ratio retention quantile windowed_conversion windowed_ratio windowed_retention windowed_quantile", "*", "winsor_fixed winsor_percentile", "*"),
        box("REF:frame.metric_unknown_type", "total active windowed_total windowed_active", "*", "none cuped winsor_fixed winsor_percentile sequential ni_margin", "*"),
        box("REF:source.frame.switchback.metric#method", "mean conversion windowed_mean windowed_conversion", "*", "cuped", "error"),
        box("REF:source.frame.switchback.metric#missing", "mean conversion windowed_mean windowed_conversion", "*", "none cuped sequential ni_margin", "zero drop"),
        box("REF:source.frame.switchback.metric#window", "windowed_mean windowed_conversion", "*", "none sequential ni_margin", "error"),
        box("REF:source.frame.switchback.metric#winsor", "mean windowed_mean", "*", "winsor_fixed winsor_percentile", "error zero drop"),
        box("REF:source.frame.switchback.plan#inference", "mean conversion", "*", "sequential", "error"),
        box("REF:source.frame.switchback.plan#margin", "mean conversion", "*", "ni_margin", "error"),
        box("RUNS", "mean conversion", "run", "none", "error"),
        box("REF:frame.metric.missing_impute", "mean conversion ratio retention quantile windowed_mean windowed_conversion windowed_ratio", "*", "*", "impute"),
        box("REF:source.frame.switchback.metric#type", "ratio retention quantile windowed_ratio", "*", "none cuped sequential ni_margin", "error zero drop"),
        box("EXC:TypeError", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "*", "cluster", "*"),
        box("REF:source.frame.switchback.identification", "mean conversion ratio retention quantile total active windowed_mean windowed_conversion windowed_ratio windowed_total windowed_active", "*", "observational", "*"),
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
    verdicts = {}
    for ingress in INGRESSES:
        rule = next((b for b in RULES[ingress] if b.matches(cell)), None)
        if rule is None:
            raise LookupError(f"no {ingress} rule classifies {cell.id}")
        verdicts[ingress] = verdict(ingress, rule.why)
    return Disposition(verdicts)


def check_disposition(cell: Cell, disposition: Disposition) -> None:
    """Raise ``AssertionError`` when a disposition breaks the matrix's own contract.

    A split between an ingress that runs and one that refuses must be explained by the
    refuser's status; construction-limited refusals of one hazard must carry one code,
    within a declaration surface and across every ingress at request stage.
    """
    refusers = {n: v for n, v in disposition.verdicts.items() if not isinstance(v.outcome, Runs)}
    if disposition.runners:
        for name, v in refusers.items():
            assert v.status in _EXPLAINS_SPLIT, (
                f"{cell.id}: {name} {v.status} cannot explain a split"
            )
    request_codes = _codes(refusers.values(), "request")
    assert len(request_codes) <= 1, (
        f"{cell.id}: one hazard, several request-stage codes {request_codes}"
    )
    for family in (_WAREHOUSE, _FRAMES):
        codes = _codes((disposition.verdicts[n] for n in family), "declaration")
        assert len(codes) <= 1, f"{cell.id}: one declaration surface, several codes {codes}"


def _codes(verdicts: Iterable[Verdict], stage: Stage) -> set[str]:
    """The refusal codes of the construction-limited verdicts raised at *stage*."""
    return {
        v.outcome.code
        for v in verdicts
        if v.status == "construction_limited"
        and v.stage == stage
        and isinstance(v.outcome, Refuses)
    }


_EXPLAINS_SPLIT = frozenset(
    {"source_limited", "not_expressible", "construction_limited", "unfinished", "unsound"}
)
