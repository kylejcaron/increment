"""Evidence-backed presentation rows for the entry-point capability reference.

Every cell names the behavioral contract that measures it: a leg of the
enumerated parity matrix, the declared disposition of a registered parity
case, or a consumer expectation read by the test that asserts it. The catalog
holds presentation text and those references only; outcomes are resolved from
the contracts when a table is rendered, never restated here.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from tests.capability_consumers_auxiliary import (
    CONSUMER_CONTRACTS as AUXILIARY_CONSUMER_CONTRACTS,
)
from tests.capability_consumers_scalar import CONSUMER_CONTRACTS as SCALAR_CONSUMER_CONTRACTS
from tests.capability_contracts import CONSTRUCTORS, ConsumerContract, ParityDeclaration
from tests.doc_evidence import resolve_evidence_function
from tests.parity_harness.declarations import PARITY_DECLARATIONS


@dataclass(frozen=True)
class MatrixBinding:
    """One leg of one parity matrix cell on one ingress."""

    cell_id: str
    ingress: str
    leg: str = "rows"


@dataclass(frozen=True)
class CaseBinding:
    """The declared disposition of a registered parity case on one ingress."""

    case_id: str
    ingress: str


@dataclass(frozen=True)
class ConsumerBinding:
    """A consumer expectation owned by the test that asserts it."""

    contract_id: str
    outcome_id: str


@dataclass(frozen=True)
class UnmeasuredBinding:
    """A cell no contract measures; the reason is displayed, never an outcome."""

    reason: str


Binding = MatrixBinding | CaseBinding | ConsumerBinding | UnmeasuredBinding

Table = Literal["ingresses", "triggered-auxiliary", "stored-state"]


@dataclass(frozen=True)
class CellClause:
    condition: str
    binding: Binding


@dataclass(frozen=True)
class EntryRow:
    """``family_clauses`` carry the conditions a capability's own family measures
    outside every Analysis constructor; they render beneath the row, never in a
    constructor column."""

    id: str
    label: str
    table: Table
    bindings: Mapping[str, tuple[CellClause, ...]]
    notes: Mapping[str, str]
    family_clauses: tuple[CellClause, ...] = ()


@dataclass(frozen=True)
class CellDisplay:
    """A resolved outcome: ``status`` is the outcome kind, ``classification`` the
    documented limitation class (SOURCE, CONSTRUCTION, COMBINATION, UNFINISHED,
    UNSOUND) when the contract records one."""

    status: str
    codes: tuple[str, ...] = ()
    reason: str = ""
    operation: str = ""
    condition: str = ""
    stage: str = ""
    classification: str = ""
    tracker: str = ""


S = "from_unit_summary"
P = "from_unit_panel"
SB = "from_switchback_panel"
D = "from_definitions"
A = "from_unit_day_artifact"
M = "from_moments"
OTHER = "other_entry_points"

INGRESSES = (S, P, SB, D, A, M)
OTHER_INGRESSES = (S, P, SB, M)

TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "ingresses": INGRESSES,
    "triggered-auxiliary": (D, A, OTHER),
    "stored-state": INGRESSES,
}

TABLE_HEADERS: Mapping[str, tuple[str, ...]] = {
    "ingresses": (
        "Unit summary",
        "Unit panel",
        "Switchback panel",
        "Definitions",
        "Unit-day artifact",
        "Portable moments",
    ),
    "triggered-auxiliary": ("Definitions", "Unit-day artifact", "Other entry points"),
    "stored-state": (
        "Unit summary",
        "Unit panel",
        "Switchback panel",
        "Definitions",
        "Unit-day artifact",
        "Portable moments",
    ),
}

_NO_DISPOSITION = "no recorded disposition"

_MATRIX_CLASSIFICATION = {
    "source_limited": "SOURCE",
    "construction_limited": "CONSTRUCTION",
    "unfinished": "UNFINISHED",
    "unsound": "UNSOUND",
}
_WAIVER_PREFIXES = ("SOURCE", "CONSTRUCTION", "COMBINATION")

_WEIGHT_DIAGNOSTIC_CONDITIONS = (
    ("IPTW, mean", "observational_iptw_covariate_adjusted_ate"),
    ("IPTW, conversion", "observational_iptw_conversion_diagnostics"),
    ("AIPW, mean", "observational_aipw_dml_covariate_adjusted"),
    ("AIPW, conversion", "observational_aipw_conversion_diagnostics"),
)


def resolve_binding(binding: Binding) -> CellDisplay:
    """Resolve a presentation clause from its named behavioral contract."""
    if isinstance(binding, UnmeasuredBinding):
        reason = binding.reason.strip()
        if not reason:
            raise ValueError("an unmeasured cell needs an explicit reason")
        if reason.casefold() == _NO_DISPOSITION:
            return CellDisplay("no_disposition")
        return CellDisplay("not_measured", reason=reason)
    if isinstance(binding, ConsumerBinding):
        expectation = _consumer_expectation(binding)
        outcome = expectation.expected
        return CellDisplay(
            outcome.kind,
            outcome.codes,
            outcome.reason,
            expectation.operation,
            expectation.condition,
            outcome.stage,
            _MATRIX_CLASSIFICATION.get(outcome.kind, ""),
        )
    if isinstance(binding, MatrixBinding):
        return _display_matrix_leg(binding)
    try:
        declaration = PARITY_DECLARATIONS[binding.case_id]
    except KeyError as exc:
        raise KeyError(f"unknown parity case {binding.case_id!r}") from exc
    return _display_parity_declaration(declaration, binding.ingress)


@cache
def _consumer_contracts() -> Mapping[str, ConsumerContract]:
    contracts: dict[str, ConsumerContract] = {}
    for contract in (*SCALAR_CONSUMER_CONTRACTS, *AUXILIARY_CONSUMER_CONTRACTS):
        if contract.id in contracts:
            raise ValueError(f"duplicate consumer contract id {contract.id!r}")
        contracts[contract.id] = contract
    return contracts


def _consumer_expectation(binding: ConsumerBinding):
    try:
        return _consumer_contracts()[binding.contract_id].outcome(binding.outcome_id)
    except KeyError as exc:
        raise KeyError(
            f"unknown consumer evidence {binding.contract_id}:{binding.outcome_id}"
        ) from exc


def _display_matrix_leg(binding: MatrixBinding) -> CellDisplay:
    # The matrix is static metadata; it is imported here so that a catalog
    # without matrix-backed rows never loads it.
    from tests.parity_harness.matrix import Refuses, StructuralAbsence, classify

    disposition = classify(_matrix_cell(binding.cell_id))
    try:
        verdict = disposition.legs[binding.leg][binding.ingress]
    except KeyError as exc:
        raise KeyError(f"unknown parity matrix leg {binding!r}") from exc
    outcome = verdict.outcome
    classification = _MATRIX_CLASSIFICATION.get(verdict.status, "")
    tracker = verdict.tracker or ""
    if isinstance(outcome, Refuses):
        return CellDisplay(
            "refused",
            (outcome.code,),
            verdict.reason,
            stage=verdict.stage,
            classification=classification,
            tracker=tracker,
        )
    if isinstance(outcome, StructuralAbsence):
        return CellDisplay(
            "not_expressible",
            reason=verdict.reason,
            stage=verdict.stage,
            classification=classification,
            tracker=tracker,
        )
    return CellDisplay("supported", reason=verdict.reason, stage=verdict.stage)


@cache
def _matrix_cell(cell_id: str):
    from tests.parity_harness.matrix import iter_asymptotic_retention_cells, iter_cells

    for cell in (*iter_cells(), *iter_asymptotic_retention_cells()):
        if cell.id == cell_id:
            return cell
    raise KeyError(f"unknown parity matrix cell {cell_id!r}")


def _split_waiver(text: str) -> tuple[str, str]:
    prefix, separator, rest = text.partition(":")
    if separator and prefix.strip() in _WAIVER_PREFIXES:
        return prefix.strip(), rest.strip()
    return "", text.strip()


def _display_parity_declaration(declaration: ParityDeclaration, ingress: str) -> CellDisplay:
    if ingress not in CONSTRUCTORS:
        raise KeyError(f"unknown ingress {ingress!r}")
    waived = dict(declaration.waive)
    refusals = dict(declaration.refusals)
    absences = {name: (error, field) for name, error, field in declaration.absences}
    warnings = dict(declaration.warnings)
    classification, reason = _split_waiver(waived.get(ingress, ""))
    if ingress in refusals:
        return CellDisplay("refused", (refusals[ingress],), reason, classification=classification)
    if ingress in absences:
        error, field = absences[ingress]
        return CellDisplay(
            "not_expressible",
            reason=f"{reason} ({error} names {field})" if reason else f"{error} names {field}",
            classification=classification,
        )
    if ingress in declaration.attempted:
        return CellDisplay("supported", warnings.get(ingress, ()))
    if ingress in waived:
        return CellDisplay("not_comparable", reason=reason, classification=classification)
    raise ValueError(f"parity declaration has no disposition for {ingress!r}")


def resolve_cell(clauses: Sequence[CellClause]) -> tuple[tuple[str, CellDisplay], ...]:
    """Resolve every clause of a cell; a consumer clause without its own
    condition is labelled by the owning expectation's condition."""
    if not clauses:
        raise ValueError("a displayed cell needs at least one clause")
    resolved: list[tuple[str, CellDisplay]] = []
    for clause in clauses:
        display = resolve_binding(clause.binding)
        condition = clause.condition
        if not condition and isinstance(clause.binding, ConsumerBinding):
            condition = display.condition
        resolved.append((condition, display))
    seen: dict[str, CellDisplay] = {}
    for condition, display in resolved:
        previous = seen.get(condition)
        if previous is not None and previous != display:
            raise ValueError(
                f"conflicting outcomes for condition {condition!r}: {previous!r} != {display!r}"
            )
        seen[condition] = display
    return tuple(resolved)


# --- row construction -----------------------------------------------------------


def _case(case_id: str, ingress: str, condition: str = "") -> CellClause:
    return CellClause(condition, CaseBinding(case_id, ingress))


def _matrix(cell_id: str, ingress: str, leg: str = "rows", condition: str = "") -> CellClause:
    return CellClause(condition, MatrixBinding(cell_id, ingress, leg))


def _consumer(contract_id: str, outcome_id: str, condition: str = "") -> CellClause:
    return CellClause(condition, ConsumerBinding(contract_id, outcome_id))


def _unmeasured(reason: str = _NO_DISPOSITION) -> CellClause:
    return CellClause("", UnmeasuredBinding(reason))


def _cases(ingress: str, *conditions: tuple[str, str]) -> tuple[CellClause, ...]:
    """One clause per ``(condition, case_id)`` on *ingress*."""
    return tuple(_case(case_id, ingress, condition) for condition, case_id in conditions)


def _row(
    row_id: str,
    label: str,
    cells: Mapping[str, Sequence[CellClause]],
    *,
    table: Table = "ingresses",
    notes: Mapping[str, str] | None = None,
    family: Sequence[CellClause] = (),
) -> EntryRow:
    return EntryRow(
        row_id,
        label,
        table,
        {column: tuple(cells[column]) for column in TABLE_COLUMNS[table]},
        dict(notes or {}),
        tuple(family),
    )


def _case_row(
    row_id: str,
    label: str,
    case_id: str,
    *,
    table: Table = "ingresses",
    notes: Mapping[str, str] | None = None,
    family: Sequence[CellClause] = (),
    **cells: Sequence[CellClause],
) -> EntryRow:
    """Every column reads the case's declared disposition unless overridden."""
    return _row(
        row_id,
        label,
        {column: cells.get(column, (_case(case_id, column),)) for column in TABLE_COLUMNS[table]},
        table=table,
        notes=notes,
        family=family,
    )


def _conditional_case_row(
    row_id: str,
    label: str,
    conditions: Sequence[tuple[str, str]],
    *,
    notes: Mapping[str, str] | None = None,
    **cells: Sequence[CellClause],
) -> EntryRow:
    """Every column reads each ``(condition, case_id)`` disposition unless overridden."""
    return _row(
        row_id,
        label,
        {column: cells.get(column, _cases(column, *conditions)) for column in INGRESSES},
        notes=notes,
    )


def _matrix_row(
    row_id: str, label: str, cell_id: str, *, leg: str = "rows", **cells: Sequence[CellClause]
) -> EntryRow:
    return _row(
        row_id,
        label,
        {column: cells.get(column, (_matrix(cell_id, column, leg),)) for column in INGRESSES},
    )


def _other_auxiliary_bindings(operation: str) -> tuple[tuple[str, str], ...]:
    return (
        (S, f"{operation}_summary"),
        (P, f"{operation}_panel"),
        (SB, f"{operation}_switchback"),
        (M, f"{operation}_moments"),
    )


def _triggered_auxiliary_row(
    row_id: str,
    label: str,
    definitions: Sequence[str],
    artifact: Sequence[str],
    other: Sequence[tuple[str, str]],
) -> EntryRow:
    """*other* pairs a constructor label with the leaf measured on that constructor
    (or one shared leaf labelled by nothing when every constructor reads alike)."""
    contract_id = "triggered_auxiliary_reads"
    return _row(
        row_id,
        label,
        {
            D: tuple(_consumer(contract_id, outcome) for outcome in definitions),
            A: tuple(_consumer(contract_id, outcome) for outcome in artifact),
            OTHER: tuple(_consumer(contract_id, outcome, label) for label, outcome in other),
        },
        table="triggered-auxiliary",
    )


# The sequential-plan gate on the switchback panel, measured on a sequential cell.
_SWITCHBACK_SEQUENTIAL_PLAN = "conversion-run-sequential-utc-error"
# The identification gate on the switchback panel, measured on an observational cell.
_SWITCHBACK_OBSERVATIONAL = "mean-run-observational-utc-error"
# An Encouragement design passed as the switchback panel's identification, measured
# by its owning readout test.
_ENCOURAGEMENT_SWITCHBACK = (
    _consumer(
        "encouragement_switchback_boundary", "switchback_encouragement_identification_refused"
    ),
)
_ENCOURAGEMENT_SEQUENTIAL_SWITCHBACK = (
    _consumer(
        "encouragement_switchback_boundary",
        "switchback_encouragement_identification_refused",
        "declared design",
    ),
    _matrix(_SWITCHBACK_SEQUENTIAL_PLAN, SB, condition="sequential plan"),
)

_PRIOR_CONDITIONS = (
    ("Normal prior", "lift_prior_normal_mean_ratio_conversion_cuped"),
    ("Student-t prior", "lift_prior_student_t_mean_ratio_conversion_cuped"),
    ("mixture prior", "lift_prior_mixture_mean_ratio_conversion_cuped"),
)
_WINSOR_CONDITIONS = (
    ("positive pool below the size route", "winsor-percentile-three-arm"),
    ("zero-inclusive pool below the size route", "winsor-percentile-zero-inclusive-2-arm"),
    ("zero-inclusive pool on the analytic size route", "winsor-percentile-size-routed-analytic"),
)
_SEGMENTED_FAMILY_CONDITIONS = (
    ("automatic levels, continuous metric", "sequential_automatic_breakout_continuous_bonferroni"),
    ("automatic levels, discrete metric", "sequential_automatic_breakout_discrete_bh"),
    ("explicit roster, continuous metric", "sequential_registered_breakout_continuous_bonferroni"),
    ("explicit roster, discrete metric", "sequential_registered_breakout_discrete_bh"),
)
_BOOLEAN_SEGMENT_CONDITIONS = tuple(
    (f"{registration} roster, {law} law", f"sequential_boolean_segments_{registration}_{law}")
    for registration in ("automatic", "registered")
    for law in ("exact_bernoulli", "asymptotic_mean")
)
_TRIGGERED_HISTORY_CONDITIONS = (
    ("daily values", "triggered_daily"),
    ("daily lift", "triggered_daily_lift"),
    ("as-of values", "triggered_asof"),
    ("as-of lift", "triggered_asof_lift"),
)
_DAY_AXIS_READS = (
    ("daily values", "run_daily"),
    ("daily lift", "run_daily_lift"),
    ("as-of values", "run_asof"),
    ("as-of lift", "run_asof_lift"),
)


def _day_axis_boundary(ingress: str) -> tuple[CellClause, ...]:
    """The triggered day-axis reads an ingress without trigger membership refuses."""
    return tuple(
        _consumer("triggered_day_axis_boundary", f"day_axis_{ingress}_{read}_refused", label)
        for label, read in _DAY_AXIS_READS
    )


_MULTI_ARM_RETENTION_UNMEASURED = "no parity case exercises multi-arm retention monitoring"
_RETENTION_CATALOG_READS = {
    ingress: (
        _case(
            "audit-compliance-only-sequential-unbounded-retention-catalog",
            ingress,
            condition="run(estimands=('compliance',)) with the retention catalog retained",
        ),
        *(
            _consumer("retained_retention_checkpoint_reads", f"{ingress}_{outcome}")
            for outcome in (
                "compliance_asof_supported",
                "itt_asof_refused",
                "itt_run_refused",
            )
        ),
    )
    for ingress in (D, A, M)
}

_INGRESS_ROWS: tuple[EntryRow, ...] = (
    _case_row(
        "mean-cuped-fixed-horizon",
        "Mean CUPED, fixed horizon",
        "mean_ratio_conversion_cuped_fixed_horizon",
        from_switchback_panel=(_matrix("mean-run-cuped-utc-error", SB),),
    ),
    _case_row(
        "ratio-cuped-fixed-horizon",
        "Ratio CUPED, fixed horizon",
        "mean_ratio_conversion_cuped_fixed_horizon",
        from_switchback_panel=(_matrix("ratio-run-cuped-utc-error", SB),),
    ),
    _row(
        "ratio-denominator-advisory-fixed-horizon",
        "Ratio denominator-precision advisory, fixed horizon",
        {
            **{
                ingress: (
                    _consumer("ratio-denominator-advisory-fixed-horizon", f"{ingress}-advisory"),
                )
                for ingress in (S, P, D, A, M)
            },
            SB: (_matrix("ratio-run-none-utc-error", SB),),
        },
    ),
    _conditional_case_row(
        "informative-log-lift-priors",
        "Informative Normal/Student-t/mixture log-lift priors, fixed-horizon "
        "mean/ratio/conversion/mean CUPED",
        _PRIOR_CONDITIONS,
    ),
    _case_row(
        "fixed-threshold-winsor-sequential",
        "Fixed-threshold winsorization under sequential inference",
        "scalar_sequential_fixed_threshold_winsor",
        from_unit_summary=(
            _case("scalar_sequential_fixed_threshold_winsor", S),
            _consumer("fixed-threshold-winsor-sequential", "unit-summary-clipped-scalars"),
        ),
        from_switchback_panel=(_matrix("mean-run-winsor_fixed-utc-error", SB),),
        family=(_consumer("fixed-threshold-winsor-sequential", "shared-fixed-threshold-gate"),),
        notes={M: "replays the exported checkpoint"},
    ),
    _conditional_case_row(
        "pooled-percentile-winsor-routes",
        "Pooled percentile winsorization, fixed horizon, default size route",
        _WINSOR_CONDITIONS,
        from_switchback_panel=(_matrix("mean-run-winsor_percentile-utc-error", SB),),
        notes={
            S: "zero outcomes are ordinary pool members; the executed route is recorded "
            "on the confidence set",
        },
    ),
    _case_row(
        "ratio-sequential",
        "Ratio metric under sequential inference",
        "audit-sequential-missing-zero",
        from_switchback_panel=(_matrix("ratio-run-sequential-utc-error", SB),),
        notes={M: "replays the exported checkpoint"},
    ),
    _case_row(
        "mean-cuped-sequential",
        "CUPED under sequential inference",
        "scalar_sequential_mean_cuped_asymptotic",
        from_unit_summary=(
            _case("scalar_sequential_mean_cuped_asymptotic", S),
            _consumer("mean-cuped-asymptotic-sequential", "predeclared-cuped-summary-state"),
            _consumer(
                "mean-cuped-asymptotic-sequential", "in-experiment-cuped-summary-joint-moments"
            ),
        ),
        family=(_consumer("mean-cuped-asymptotic-sequential", "exact-route-cuped-refused"),),
        notes={
            S: "asymptotic route",
            D: "asymptotic route",
            A: "asymptotic route",
            M: "replays the exported checkpoint",
        },
    ),
    _conditional_case_row(
        "ratio-cuped-sequential",
        "Ratio CUPED under sequential inference",
        (
            ("overlapping arm denominators", "audit-adjusted-ratio-anchor"),
            ("separated arm denominators", "audit-adjusted-ratio-anchor-resolved"),
        ),
        from_switchback_panel=(_matrix("ratio-run-cuped-utc-error", SB),),
        notes={
            S: "asymptotic route",
            D: "asymptotic route",
            A: "asymptotic route",
            M: "replays the exported checkpoint",
        },
    ),
    _row(
        "bounded-conversion-asymptotic-sequential",
        "Bounded conversion under asymptotic sequential inference",
        {
            **{
                ingress: (
                    _consumer(
                        "bounded-conversion-sequential-state-and-interval",
                        f"asymptotic_mean-{ingress}-retained-state",
                    ),
                )
                for ingress in (S, P, D, A, M)
            },
            SB: (_matrix(_SWITCHBACK_SEQUENTIAL_PLAN, SB),),
        },
        notes={
            S: "pre-windowed per-unit outcomes with an exposure date",
            M: "replays the exported checkpoint",
        },
    ),
    _matrix_row(
        "bounded-retention-asymptotic-sequential",
        "Bounded retention (a start and end threshold) under asymptotic sequential inference",
        "retention-run-sequential-utc-error-asymptotic_mean",
    ),
    _row(
        "bounded-conversion-exact-bernoulli",
        "Bounded conversion under automatic exact Bernoulli monitoring without a registration",
        {
            **{
                ingress: (
                    _consumer(
                        "bounded-conversion-sequential-state-and-interval",
                        f"always_valid-{ingress}-retained-state",
                    ),
                )
                for ingress in (S, P, D, A, M)
            },
            SB: (_matrix(_SWITCHBACK_SEQUENTIAL_PLAN, SB),),
        },
        notes={
            S: "pre-windowed per-unit outcomes with an exposure date",
            M: "replays the exported checkpoint",
        },
    ),
    _matrix_row(
        "bounded-retention-exact-bernoulli",
        "Bounded retention under automatic exact Bernoulli monitoring without a registration",
        "retention-run-sequential-utc-error",
    ),
    _case_row(
        "multi-arm-bounded-conversion-exact",
        "Automatic exact multi-arm bounded-conversion monitoring",
        "sequential_automatic_multi_arm_exact",
        from_switchback_panel=(
            _consumer("switchback_multi_arm_identification", "conversion_multi_arm_refused"),
            _matrix(
                _SWITCHBACK_SEQUENTIAL_PLAN,
                SB,
                condition="request reduced to control plus one treatment; sequential plan",
            ),
        ),
        notes={
            S: "pre-windowed per-unit outcomes with an exposure date",
            M: "replays the exported checkpoint",
        },
    ),
    _row(
        "multi-arm-bounded-retention-exact",
        "Automatic exact multi-arm bounded-retention monitoring",
        {
            S: (_matrix("retention-run-sequential-utc-error", S),),
            P: (_unmeasured(_MULTI_ARM_RETENTION_UNMEASURED),),
            SB: (
                _consumer("switchback_multi_arm_identification", "retention_multi_arm_refused"),
                _matrix(
                    "retention-run-sequential-utc-error",
                    SB,
                    condition="request reduced to control plus one treatment; retention metric",
                ),
            ),
            D: (_unmeasured(_MULTI_ARM_RETENTION_UNMEASURED),),
            A: (_unmeasured(_MULTI_ARM_RETENTION_UNMEASURED),),
            M: (_unmeasured(_MULTI_ARM_RETENTION_UNMEASURED),),
        },
    ),
    _matrix_row(
        "unbounded-conversion-sequential",
        "Unbounded conversion under sequential inference",
        "conversion-run-sequential-utc-error",
    ),
    _row(
        "unbounded-retention-sequential",
        "Unbounded retention (an integer threshold) under sequential inference",
        {
            S: (_consumer("unbounded-retention-sequential", "summary-no-dates-refused"),),
            P: (_unmeasured(),),
            SB: (_consumer("unbounded-retention-sequential", "switchback-retention-refused"),),
            D: (
                _consumer(
                    "unbounded-retention-sequential", "definitions-unbounded-observation-refused"
                ),
            ),
            A: (_unmeasured(),),
            M: (_consumer("unbounded-retention-sequential", "moments-no-checkpoint-to-replay"),),
        },
    ),
    _case_row(
        "unit-summary-whole-window-breakouts",
        "Whole-window breakouts from one-row-per-unit summaries",
        "breakout_store_dimension",
        notes={
            S: "breakouts do not create daily or as-of capabilities",
            M: "portable moments contain no per-unit segment labels",
        },
    ),
    _conditional_case_row(
        "registered-segmented-sequential-family",
        "Randomized registered segmented sequential family (automatic predeclared levels "
        "or explicit roster)",
        _SEGMENTED_FAMILY_CONDITIONS,
        from_switchback_panel=(_matrix("mean-breakout-sequential-utc-error", SB),),
        notes={M: "replays the retained segment state; a cube carries no breakout catalog"},
    ),
    _conditional_case_row(
        "boolean-null-segment-labels",
        "Boolean and null segment labels in a registered segmented sequential family "
        "(canonical true/false/null spellings matching the string-labelled oracle)",
        _BOOLEAN_SEGMENT_CONDITIONS,
        from_switchback_panel=(_matrix("conversion-breakout-sequential-utc-error", SB),),
        notes={
            S: "measured on pandas, Polars and Arrow Boolean columns",
            P: "measured on pandas, Polars and Arrow Boolean columns",
            M: "replays the retained segment state; a cube carries no breakout catalog",
        },
    ),
    _row(
        "triggered-sequential-registration",
        "Registered sequential monitoring of the triggered population "
        "(Experiment.trigger with InferenceSpec; assigned and triggered chains)",
        {
            **{
                ingress: (_consumer("triggered-sequential-boundary", ingress),)
                for ingress in (S, P, SB)
            },
            D: (
                _case(
                    "triggered_sequential_asymptotic_mean", D, "registered plan and certified feeds"
                ),
                _consumer("triggered-sequential-certification", "native-feed"),
                _consumer("triggered-sequential-certification", "native-sql"),
                _consumer("triggered-sequential-certification", "native-unpinned"),
            ),
            A: (
                _case(
                    "triggered_sequential_asymptotic_mean",
                    A,
                    "trigger extensions and certified feeds",
                ),
                _consumer("triggered-sequential-certification", "artifact-feed"),
            ),
            M: (
                _case("triggered_sequential_asymptotic_mean", M, "retained triggered checkpoint"),
                _consumer("triggered-sequential-boundary", M),
            ),
        },
        notes={
            A: "capture uses trigger_population, trigger_measure_stats and triggered assignment_counts; "
            "encouragement also needs encouragement_uptake",
            M: "replays retained assigned and triggered state; cannot start either process",
        },
    ),
    _row(
        "triggered-sequential-uptake",
        "Triggered encouragement compliance under a registered plan",
        {
            **{
                ingress: (_consumer("triggered-sequential-boundary", ingress),)
                for ingress in (S, P, SB)
            },
            **{
                ingress: (_consumer("triggered-sequential-uptake", f"{ingress}-unavailable"),)
                for ingress in (D, A, M)
            },
        },
        notes=dict.fromkeys((D, A, M), "assigned compliance remains monitored"),
    ),
    _row(
        "impute-pooled-mean-roles",
        "Pooled-mean imputation helper with declared roles; a frame metric declaring "
        "an impute missing policy",
        {
            **{
                ingress: (
                    _consumer(
                        "impute-pooled-mean-role-and-metric-policies",
                        f"{ingress}-outcome-role-refused",
                    ),
                    _consumer(
                        "impute-pooled-mean-role-and-metric-policies",
                        f"{ingress}-undeclared-role-warning",
                    ),
                    _consumer(
                        "impute-pooled-mean-role-and-metric-policies",
                        f"{ingress}-missing-impute-refused",
                    ),
                )
                for ingress in (S, P, SB)
            },
            D: (
                _consumer(
                    "impute-pooled-mean-role-and-metric-policies",
                    "from_definitions-missing-policy-not-declarable",
                ),
                _matrix("mean-run-none-utc-impute", D, condition="impute missing policy"),
            ),
            A: (
                _consumer(
                    "impute-pooled-mean-role-and-metric-policies",
                    "from_unit_day_artifact-missing-policy-not-declarable",
                ),
                _matrix("mean-run-none-utc-impute", A, condition="impute missing policy"),
            ),
            M: (
                _consumer(
                    "impute-pooled-mean-role-and-metric-policies", "from_moments-no-per-unit-rows"
                ),
                _matrix("mean-run-none-utc-impute", M, condition="impute missing policy"),
            ),
        },
    ),
    _case_row(
        "encouragement-design-readouts",
        "Encouragement ITT, compliance and LATE, fixed horizon, declared as the study design",
        "encouragement_declared_definitions",
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
    ),
    _conditional_case_row(
        "encouragement-no-exclusion",
        "Explicit ITT/compliance without a LATE exclusion declaration (fixed horizon or "
        "sequential monitoring)",
        (
            ("fixed horizon", "encouragement_optional_exclusion"),
            ("sequential monitoring", "sequential_encouragement_optional_exclusion"),
        ),
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
        notes={M: "sequential monitoring replays the exported checkpoint"},
    ),
    _case_row(
        "encouragement-empty-outcome-catalog",
        "Fixed-horizon design-level compliance with an empty outcome catalog",
        "audit-compliance-only-empty-catalog",
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
        notes={M: "a design summary carrying the source identity and the empty catalog"},
    ),
    _conditional_case_row(
        "encouragement-margin-guardrail",
        "Encouragement guardrail with a declared non-inferiority margin",
        (
            ("request includes itt", "encouragement_margin_guardrail"),
            ("request excludes itt", "encouragement_margin_guardrail_late_only"),
        ),
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
        notes=dict.fromkeys((S, P, D, A, M), "the ITT row carries the margin verdict"),
    ),
    _case_row(
        "encouragement-late-fcr-reestimated",
        "Encouragement BH-selected secondary's LATE row re-estimated at the corrected level",
        "encouragement_fcr_reestimation",
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
    ),
    _conditional_case_row(
        "encouragement-itt-conversion-route",
        "Encouragement ITT on a conversion metric takes the count-routed conversion route",
        (
            (
                "dense counts, delta-method route",
                "encouragement_itt_dense_takes_the_delta_method_route",
            ),
            (
                "sparse counts, finite-sample route",
                "encouragement_itt_keeps_binomial_reference_kind",
            ),
        ),
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
    ),
    _case_row(
        "conversion-route-threshold",
        "Conversion/retention counts one below, at and one above the dense threshold, on "
        "the success and failure side, route identically and report the same interval and "
        "p-value",
        "conversion_route_straddles_the_dense_threshold",
    ),
    _case_row(
        "conversion-breakout-route",
        "Conversion/retention breakout with dense, sparse and threshold-straddling segments "
        "in one metric: each segment row is routed by its own counts",
        "conversion_route_breakout_segments",
        from_moments=(_matrix("conversion-breakout-none-utc-error", M),),
    ),
    _case_row(
        "conversion-route-encouragement-family",
        "Encouragement ITT rows of a BH secondary family are routed at their own level and carry "
        "composite p-value evidence",
        "conversion_route_encouragement_family",
        from_switchback_panel=_ENCOURAGEMENT_SWITCHBACK,
    ),
    _case_row(
        "conversion-route-observational-family",
        "Observational unadjusted rows of a BH secondary family are routed at their own level and "
        "carry composite p-value evidence",
        "conversion_route_observational_family",
        from_switchback_panel=(_matrix("conversion-run-observational-utc-error", SB),),
    ),
    _row(
        "encouragement-retention-readouts",
        "Encouragement design with a retention metric (ITT and LATE readouts)",
        {
            S: (_consumer("encouragement_retention_readouts", "itt_late_summary_refused"),),
            P: (_consumer("encouragement_retention_readouts", "itt_late_panel_refused"),),
            SB: _ENCOURAGEMENT_SWITCHBACK,
            D: (_consumer("encouragement_retention_readouts", "itt_late_definitions_refused"),),
            A: (_consumer("encouragement_retention_readouts", "itt_late_artifact_refused"),),
            M: (_consumer("encouragement_retention_readouts", "itt_late_moments_refused"),),
        },
    ),
    _row(
        "encouragement-retention-compliance-only",
        "Encouragement design with a retention metric, compliance-only request "
        "(which ignores the retention outcome)",
        {
            S: (_consumer("encouragement_retention_readouts", "itt_late_summary_refused"),),
            P: (_consumer("encouragement_retention_readouts", "itt_late_panel_refused"),),
            SB: _ENCOURAGEMENT_SWITCHBACK,
            D: (
                _consumer(
                    "encouragement_retention_readouts", "compliance_only_definitions_supported"
                ),
                _consumer("encouragement_retention_readouts", "compliance_only_itt_still_refused"),
            ),
            A: (
                _unmeasured(
                    "no consumer exercises a compliance-only request on an adopted artifact"
                ),
            ),
            M: (_unmeasured("no consumer exercises a compliance-only request on a replayed cube"),),
        },
        notes={
            S: "the retention declaration itself is the limit, before any request",
            P: "the retention declaration itself is the limit, before any request",
        },
    ),
    _case_row(
        "encouragement-conversion-breakout-zero-control",
        "Encouragement conversion ITT breakouts retain exact confidence sets at zero "
        "control counts",
        "audit-binary-breakout-ancillary-uptake",
    ),
    _case_row(
        "retention-breakout-cohorts",
        "Bounded-retention breakouts with mature exposure cohorts",
        "audit-retention-breakout-cohorts",
        notes={
            P: "breakout rows",
            D: "breakout rows and cohort-aware breakout summaries",
            A: "breakout rows and cohort-aware breakout summaries",
        },
    ),
    _conditional_case_row(
        "encouragement-joint-sequential",
        "Encouragement continuous ITT + Bernoulli uptake compliance, jointly monitored "
        "(composed sequential family, e-BH selected)",
        (
            ("ITT and uptake only", "sequential_composed_itt_and_uptake"),
            ("with in-family secondaries", "sequential_composed_family_ebh"),
        ),
        from_switchback_panel=_ENCOURAGEMENT_SEQUENTIAL_SWITCHBACK,
        notes={M: "replays the exported checkpoint"},
    ),
    _case_row(
        "uptake-only-sequential-empty-catalog",
        "Uptake-only sequential compliance with an empty outcome catalog",
        "audit-compliance-only-sequential-empty-catalog",
        from_switchback_panel=_ENCOURAGEMENT_SEQUENTIAL_SWITCHBACK,
        notes={M: "replays the exported checkpoint with the empty catalog"},
    ),
    _case_row(
        "uptake-only-sequential-unbounded-retention-catalog",
        "Uptake-only sequential compliance that retains an unbounded retention metric in "
        "the outcome catalog",
        "audit-compliance-only-sequential-unbounded-retention-catalog",
        from_switchback_panel=_ENCOURAGEMENT_SEQUENTIAL_SWITCHBACK,
        from_definitions=_RETENTION_CATALOG_READS[D],
        from_unit_day_artifact=_RETENTION_CATALOG_READS[A],
        from_moments=_RETENTION_CATALOG_READS[M],
        notes={
            A: "the retention metric declared on the experiment and its catalog are retained",
            M: "replays the exported checkpoint with the retention catalog retained",
        },
    ),
    _conditional_case_row(
        "observational-weight-diagnostics",
        "Observational weight diagnostics (IPTW/AIPW; unit-level mean and conversion parity)",
        _WEIGHT_DIAGNOSTIC_CONDITIONS,
        from_unit_summary=_cases(S, *_WEIGHT_DIAGNOSTIC_CONDITIONS)
        + (
            _consumer(
                "observational_cluster_weight_diagnostics",
                "summary_cluster_weight_grain",
            ),
        ),
        from_definitions=_cases(D, *_WEIGHT_DIAGNOSTIC_CONDITIONS)
        + (
            _consumer(
                "observational_cluster_weight_diagnostics",
                "definitions_cluster_weight_grain",
            ),
        ),
        from_switchback_panel=(
            _matrix(_SWITCHBACK_OBSERVATIONAL, SB, condition="mean"),
            _matrix("conversion-run-observational-utc-error", SB, condition="conversion"),
        ),
    ),
    _case_row(
        "assignment-integrity-scope",
        "Assignment-integrity scope (status, observed counts, randomization grain and "
        "retained metadata; triggered populations are not assignment-law evidence)",
        "clustered_assignment_integrity_scope",
        from_unit_summary=(
            _case("clustered_assignment_integrity_scope", S, "cluster-grain scope"),
            _consumer("assignment_integrity_source_grain", "summary_cluster_grain_supported"),
        ),
        from_unit_panel=(
            _consumer("assignment_integrity_source_grain", "panel_unit_grain_supported"),
            _case("clustered_assignment_integrity_scope", P, "cluster grain"),
        ),
        from_moments=(
            _consumer("assignment_integrity_source_grain", "moments_unit_grain_supported"),
            _case("clustered_assignment_integrity_scope", M, "cluster grain"),
        ),
        notes={
            S: "unassigned totals are separate",
            D: "unit or cluster-grain counts with mixed/unassigned audit totals",
            A: "cluster grain requires cluster identity; mixed/unassigned totals require "
            "assignment-count evidence",
        },
    ),
    _case_row(
        "observational-iptw-mean",
        "Observational IPTW covariate adjustment",
        "observational_iptw_covariate_adjusted_ate",
        from_switchback_panel=(_matrix(_SWITCHBACK_OBSERVATIONAL, SB),),
    ),
    _case_row(
        "observational-aipw-dml",
        "Observational AIPW/DML covariate adjustment",
        "observational_aipw_dml_covariate_adjusted",
        from_switchback_panel=(_matrix(_SWITCHBACK_OBSERVATIONAL, SB),),
    ),
    _case_row(
        "categorical-observational-adjustment",
        "Categorical observational adjustment (string covariates)",
        "observational_categorical_covariate_adjusted",
        from_switchback_panel=(_matrix(_SWITCHBACK_OBSERVATIONAL, SB),),
    ),
    _case_row(
        "categorical-missing-policy",
        "Categorical nulls with a nondefault missing-value policy",
        "observational_categorical_covariate_missing_levels",
        from_switchback_panel=(_matrix(_SWITCHBACK_OBSERVATIONAL, SB),),
        notes={
            D: "the definitions grammar declares no missing-covariate policy, so the default "
            "applies",
            A: "the default missing-covariate policy is inherited from definitions",
        },
    ),
    _case_row(
        "observational-role-multiplicity",
        "Observational primary + secondary + guardrail together (role multiplicity)",
        "observational_multiplicity_primary_secondary_guardrail",
        from_switchback_panel=(_matrix(_SWITCHBACK_OBSERVATIONAL, SB),),
    ),
    _case_row(
        "quantile-tied-rounded-outcomes",
        "Quantile inference on tied/rounded outcomes",
        "quantile_tied_rounded_outcomes",
        from_unit_panel=(
            _case("quantile_tied_rounded_outcomes", P, "unwindowed metric"),
            _matrix("windowed_quantile-run-none-utc-error", P, condition="windowed metric"),
        ),
        from_switchback_panel=(_matrix("quantile-run-none-utc-error", SB),),
        from_moments=(
            _case("quantile_tied_rounded_outcomes", M, "genuine quantile export"),
            _matrix(
                "quantile-run-none-utc-error", M, condition="quantile declared over a scalar cube"
            ),
            _matrix(
                "quantile-breakout-none-utc-error",
                M,
                condition="breakout over a scalar cube",
            ),
            _matrix(
                "quantile-daily-none-utc-error",
                M,
                "values",
                condition="day-axis read over a scalar cube",
            ),
        ),
        notes={
            P: "the panel's per-unit collapse is the same per-unit total mean/ratio already use"
        },
    ),
    _case_row(
        "quantile-checkpoint-retention",
        "Registered conversion checkpoint retaining an unmodeled, unwindowed quantile catalog",
        "sequential-mixed-quantile-catalog-checkpoint",
        from_moments=(
            _case("sequential-mixed-quantile-catalog-checkpoint", M, "checkpoint replay"),
            _consumer(
                "moments_checkpoint_reexport_boundary", "moments_checkpoint_reexport_refused"
            ),
        ),
        notes={
            S: "export and replay",
            P: "export and replay",
            D: "export and replay",
            A: "export and replay",
        },
    ),
    _case_row(
        "observational-quantile-refusal",
        "Quantile metric under an Observational design (two-sided, zero-null request)",
        "observational_quantile_refused_at_the_readout_seam",
        from_unit_summary=(
            _case("observational_quantile_refused_at_the_readout_seam", S, "two-sided zero null"),
            _consumer(
                "observational_relative_margin_boundary",
                "summary_absolute_margin_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "summary_one_sided_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "summary_relative_margin_quantile_precedence_refused",
            ),
        ),
        from_unit_panel=(
            _case("observational_quantile_refused_at_the_readout_seam", P, "two-sided zero null"),
            _consumer("observational_relative_margin_boundary", "panel_relative_margin_refused"),
            _consumer(
                "observational_relative_margin_boundary",
                "panel_absolute_margin_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "panel_one_sided_precedence_refused",
            ),
        ),
        from_switchback_panel=(_matrix("quantile-run-observational-utc-error", SB),),
        from_definitions=(
            _case("observational_quantile_refused_at_the_readout_seam", D, "two-sided zero null"),
            _consumer(
                "observational_relative_margin_boundary",
                "from_definitions_absolute_margin_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "from_definitions_one_sided_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "relative_margin_public_definitions_constructor_refused",
            ),
        ),
        from_unit_day_artifact=(
            _case("observational_quantile_refused_at_the_readout_seam", A, "two-sided zero null"),
            _consumer(
                "observational_relative_margin_boundary",
                "from_unit_day_artifact_absolute_margin_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "from_unit_day_artifact_one_sided_quantile_precedence_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "relative_margin_artifact_context_unexpressible",
            ),
        ),
        from_moments=(
            _case(
                "observational_quantile_refused_at_the_readout_seam",
                M,
                "two-sided zero null over a scalar cube",
            ),
            _consumer(
                "observational_relative_margin_boundary", "observational_quantile_two_sided_refused"
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "observational_quantile_alternative_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary", "observational_quantile_guardrail_refused"
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "observational_quantile_absolute_margin_refused",
            ),
            _consumer(
                "observational_relative_margin_boundary", "relative_margin_quantile_moments_refused"
            ),
            _consumer(
                "observational_relative_margin_boundary",
                "relative_margin_moments_constructor_refused",
            ),
            _matrix(
                "quantile-breakout-observational-utc-error",
                M,
                condition="breakout over a scalar cube",
            ),
            _matrix(
                "quantile-daily-observational-utc-error",
                M,
                "values",
                condition="day-axis read over a scalar cube",
            ),
        ),
        notes={M: "genuine observational quantile export raises the same estimator limit"},
    ),
    _case_row(
        "clustered-negative-mean",
        "Clustered ratio metric with a non-positive control-arm numerator",
        "clustered_negative_control_mean_produces_a_fieller_set",
        from_unit_panel=(_matrix("ratio-run-cluster-utc-error", P),),
        from_switchback_panel=(_matrix("ratio-run-cluster-utc-error", SB),),
        notes={P: "build clustered arm totals with the unit summary instead"},
    ),
    _case_row(
        "clustered-zero-relative-variance",
        "Clustered zero relative variance retains its point and any available additive interval",
        "clustered_zero_relative_variance_preserves_point_and_additive_interval",
        from_unit_panel=(_matrix("ratio-run-cluster-utc-error", P),),
        from_switchback_panel=(_matrix("ratio-run-cluster-utc-error", SB),),
        notes={P: "build clustered arm totals with the unit summary instead"},
    ),
    _row(
        "planning-baseline-mean",
        "Planning baseline, mean metric",
        {
            S: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "mean-summary"),),
            P: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "mean-panel"),),
            SB: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "mean-switchback-planner"
                ),
            ),
            D: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "mean-definitions"),),
            A: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "mean-artifact"),),
            M: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "mean-moments"),),
        },
        notes={SB: "the pilot-fitted switchback baseline feeds the switchback solvers"},
    ),
    _row(
        "planning-baseline-quantile",
        "Planning baseline, quantile metric",
        {
            S: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "quantile-summary"),),
            P: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "quantile-panel"),),
            SB: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "quantile-switchback-refused"
                ),
            ),
            D: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "quantile-triggered-definitions",
                ),
            ),
            A: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "quantile-triggered-artifact"
                ),
            ),
            M: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "quantile-moments-refusal"
                ),
                _case("quantile_tied_rounded_outcomes", M, "genuine quantile export"),
            ),
        },
        notes={P: "unwindowed metrics only, as for quantile inference on the panel"},
    ),
    _row(
        "planning-baseline-cluster",
        "Planning baseline with a declared cluster (per-unit mean/var; ICC on the metric's "
        "per-unit score)",
        {
            S: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "cluster-summary"),),
            P: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "cluster-panel-refusal"
                ),
            ),
            SB: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "cluster-switchback-absence"
                ),
            ),
            D: (
                _consumer("planning-baseline-mean-quantile-cluster-trigger", "cluster-definitions"),
            ),
            A: (_consumer("planning-baseline-mean-quantile-cluster-trigger", "cluster-artifact"),),
            M: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "cluster-moments-refusal"
                ),
            ),
        },
    ),
    _row(
        "planning-baseline-trigger",
        "Planning baseline under a declared trigger (triggered-population moments and "
        "trigger rate)",
        {
            S: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-summary-assigned-population",
                ),
            ),
            P: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-panel-assigned-population",
                ),
            ),
            SB: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-switchback-no-trigger",
                ),
            ),
            D: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-definitions-complete",
                ),
            ),
            A: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger", "trigger-artifact-complete"
                ),
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-artifact-missing-evidence",
                ),
            ),
            M: (
                _consumer(
                    "planning-baseline-mean-quantile-cluster-trigger",
                    "trigger-moments-assigned-population",
                ),
            ),
        },
        notes={
            S: "the assigned population is analyzed instead",
            A: "trigger population, triggered assignment counts and metric-scoped trigger "
            "measure statistics, with explicit source snapshot evidence",
        },
    ),
    _conditional_case_row(
        "triggered-day-axis-histories",
        "Triggered daily/as-of outcome value and lift history (membership is limited to "
        "units whose first eligible trigger is observed by each day, and outcome windows "
        "anchor at that unit's trigger)",
        _TRIGGERED_HISTORY_CONDITIONS,
        **{ingress: _day_axis_boundary(ingress) for ingress in (S, P, SB, M)},
        notes={
            D: "explicit source snapshot evidence; outcome events are strictly after each "
            "unit's first eligible trigger and are cut off by source evidence",
            A: "trigger population and metric-compatible trigger measure statistics",
        },
    ),
    _case_row(
        "triggered-encouragement-compliance",
        "Triggered encouragement compliance (daily and as-of lift, compliance only; uptake "
        "stays assignment-anchored while membership enters at first eligible trigger)",
        "triggered_encouragement_compliance_asof",
        notes={
            D: "explicit source snapshot evidence; uptake windows use assignment time and "
            "triggered membership uses the first-trigger day",
            A: "trigger population and encouragement uptake evidence",
        },
    ),
    _row(
        "triggered-clustered-day-axis",
        "Triggered clustered daily/as-of outcome readouts",
        {
            S: _day_axis_boundary(S),
            P: _day_axis_boundary(P),
            SB: _day_axis_boundary(SB),
            D: (
                _consumer("triggered_clustered_day_axis", "clustered_outcome_history_refused"),
                _consumer("triggered_clustered_day_axis", "clustered_compliance_history_supported"),
                _consumer(
                    "triggered_clustered_day_axis", "clustered_triggered_population_supported"
                ),
            ),
            A: (
                _consumer(
                    "triggered_clustered_day_axis", "clustered_artifact_outcome_history_refused"
                ),
                _consumer(
                    "triggered_clustered_day_axis",
                    "clustered_artifact_compliance_history_supported",
                ),
            ),
            M: _day_axis_boundary(M),
        },
    ),
    _row(
        "logged-policy-contrast",
        "Logged-policy contrast",
        {
            ingress: (
                _consumer(
                    "logged_policy_family_boundary", f"analysis_ingress_{ingress}_not_expressible"
                ),
            )
            for ingress in INGRESSES
        },
        family=(_consumer("logged_policy_family_boundary", "logged_trace_contrast_supported"),),
    ),
    _row(
        "tabular-policy-persistence",
        "TabularPolicy persistence (JSON dump and validate)",
        {
            ingress: (
                _consumer(
                    "logged_policy_family_boundary",
                    f"policy_persistence_analysis_ingress_{ingress}_not_comparable",
                ),
            )
            for ingress in INGRESSES
        },
        family=(
            _consumer("logged_policy_family_boundary", "tabular_policy_string_json_supported"),
            _consumer("logged_policy_family_boundary", "tabular_policy_typed_json_refused"),
            _consumer("logged_policy_family_boundary", "tabular_policy_typed_python_supported"),
        ),
    ),
)


_AUXILIARY_ROWS: tuple[EntryRow, ...] = (
    _triggered_auxiliary_row(
        "triggered-read-breakout",
        "`run_breakout()` / `breakout_summaries()` with a declared trigger",
        ("breakout_definitions_triggered_population",),
        (
            "breakout_artifact_complete_trigger_evidence",
            "breakout_artifact_missing_extension",
            "breakout_artifact_missing_evidence",
        ),
        _other_auxiliary_bindings("breakout"),
    ),
    _triggered_auxiliary_row(
        "triggered-read-factor-summaries",
        "`factor_summaries()` with a declared trigger",
        ("factor_definitions_triggered_population",),
        (
            "factor_artifact_complete_trigger_evidence",
            "factor_artifact_missing_extension",
            "factor_artifact_missing_evidence",
        ),
        _other_auxiliary_bindings("factor"),
    ),
    _triggered_auxiliary_row(
        "triggered-read-dashboard-group",
        "`dashboard_group_data(population=...)`",
        ("dashboard_definitions_population",),
        ("dashboard_artifact_refused",),
        _other_auxiliary_bindings("dashboard"),
    ),
    _triggered_auxiliary_row(
        "triggered-read-allocation-history",
        "`allocation_history(population=...)`",
        ("allocation_definitions_population",),
        ("allocation_artifact_refused",),
        (
            ("unit summary", "allocation_from_unit_summary_refused"),
            ("unit panel", "allocation_from_unit_panel_refused"),
            ("switchback panel", "allocation_from_switchback_panel_refused"),
            ("portable moments", "allocation_moments_refused"),
        ),
    ),
    _triggered_auxiliary_row(
        "triggered-read-sitewide-assigned",
        "`sitewide(population='assigned')` with a declared trigger",
        (
            "sitewide_definitions_assigned_supported",
            "sitewide_assigned_observational_refused",
            "sitewide_definitions_metric_unaligned_refused",
            "sitewide_definitions_conversion_refused",
        ),
        (
            "sitewide_artifact_assigned_supported",
            "sitewide_artifact_metric_unaligned_refused",
            "sitewide_artifact_conversion_refused",
        ),
        _other_auxiliary_bindings("sitewide"),
    ),
    _triggered_auxiliary_row(
        "triggered-read-sitewide",
        "`sitewide(population='triggered')` with a declared trigger",
        (
            "sitewide_definitions_triggered_supported",
            "sitewide_definitions_metric_unaligned_refused",
            "sitewide_definitions_conversion_refused",
        ),
        (
            "sitewide_artifact_triggered_supported",
            "sitewide_artifact_metric_unaligned_refused",
            "sitewide_artifact_conversion_refused",
        ),
        (
            ("unit summary", "sitewide_from_unit_summary_triggered_population_refused"),
            ("unit panel", "sitewide_from_unit_panel_triggered_population_refused"),
            ("switchback panel", "sitewide_from_switchback_panel_triggered_population_refused"),
            ("portable moments", "sitewide_from_moments_triggered_population_refused"),
        ),
    ),
    _triggered_auxiliary_row(
        "triggered-read-cate",
        "`estimate_cate()`, `validate_cate()`, `targeting_rule()`, `select_targeting_rule()` "
        "with a declared trigger",
        (
            "cate_definitions_triggered_population",
            "cate_definitions_observational_validation_supported",
            "cate_definitions_observational_targeting_supported",
            "cate_definitions_observational_selection_supported",
            "cate_definitions_observational_covariates_refused",
        ),
        (
            "cate_artifact_complete_trigger_evidence",
            "cate_artifact_observational_validation_supported",
            "cate_artifact_observational_targeting_supported",
            "cate_artifact_observational_selection_supported",
            "cate_artifact_observational_covariates_refused",
            "cate_artifact_missing_extension",
            "cate_artifact_missing_evidence",
        ),
        _other_auxiliary_bindings("cate"),
    ),
    _row(
        "callable-experiment-health",
        "`Analysis.health_check()` source diagnostics",
        {
            D: (_consumer("callable_experiment_health", "health_definitions_triggered"),),
            A: (
                _consumer("callable_experiment_health", "health_artifact_triggered"),
                _consumer("callable_experiment_health", "health_artifact_without_extensions"),
            ),
            OTHER: (
                _consumer("callable_experiment_health", "health_unit_summary"),
                _consumer("callable_experiment_health", "health_unit_panel"),
                _consumer("callable_experiment_health", "health_switchback"),
                _consumer("callable_experiment_health", "health_moments"),
            ),
        },
        table="triggered-auxiliary",
        notes={
            D: "matched native definitions health parity test uses explicit source cutoff and feed watermark",
            A: "matched artifact health parity requires trigger_population, assignment_counts and trigger_measure_stats",
            OTHER: "unit summary, unit panel, switchback and moments do not retain trigger population evidence",
        },
    ),
)

_BOOLEAN_CONTINUATION_CAPTURE = (
    ("exact Bernoulli law", "sequential_boolean_segments_registered_exact_bernoulli"),
    ("asymptotic mean law", "sequential_boolean_segments_registered_asymptotic_mean"),
)
_WINDOW_SPELLING_CONDITIONS = (
    ("zulu and offset spellings of one instant", "window_spelling_zulu_vs_offset"),
    ("two offset spellings of one instant", "window_spelling_offset_vs_offset"),
)

_STORED_STATE_ROWS: tuple[EntryRow, ...] = (
    _row(
        "triggered-sequential-continuation",
        "Registered process committed without a triggered chain continued under a declared trigger",
        {
            **{
                ingress: (_consumer("triggered-sequential-boundary", ingress),)
                for ingress in (S, P, SB)
            },
            **{
                ingress: (
                    _consumer("triggered-sequential-continuation", f"{ingress}-unavailable"),
                    _consumer("triggered-sequential-continuation", f"{ingress}-backfill"),
                )
                for ingress in (D, A, M)
            },
        },
        table="stored-state",
        notes={
            D: "continues the assigned chain only",
            A: "continues the assigned chain only",
            M: "replays the retained assigned-only commitment",
        },
    ),
    _row(
        "stored-boolean-null-continuation",
        "Continuing a checkpoint recorded under earlier raw Boolean/null spellings with "
        "canonically labelled units",
        {
            S: (
                _consumer("stored_bool_null_continuation", "canonical_relabel_refused"),
                _consumer("stored_bool_null_continuation", "same_spelling_continuation"),
            ),
            P: (
                _consumer("stored_bool_null_continuation", "panel_legacy_unbounded_prefix_refused"),
            ),
            SB: (_matrix(_SWITCHBACK_SEQUENTIAL_PLAN, SB),),
            D: _cases(D, *_BOOLEAN_CONTINUATION_CAPTURE),
            A: _cases(A, *_BOOLEAN_CONTINUATION_CAPTURE),
            M: (_consumer("stored_bool_null_continuation", "raw_bool_null_replay"),),
        },
        table="stored-state",
        notes={S: "the checkpoint itself replays unchanged"},
    ),
    _row(
        "artifact-context-formats",
        "Artifact context format 2 (hash-only source recipes; no source SQL)",
        {
            **{
                ingress: (_case("artifact_context_format_two", ingress),)
                for ingress in (S, P, SB, D, M)
            },
            A: (
                _case("artifact_context_format_two", A, "format 2"),
                _case("artifact_context_format_one_refusal", A, "format 1"),
            ),
        },
        table="stored-state",
        notes={
            D: "publishes format 2",
            A: "a format-1 context must be republished from trusted definitions",
        },
    ),
    _row(
        "experiment-window-day-boundary",
        "Experiment window days (start, end and observation-horizon days) follow the day "
        "boundary: a naive value is the wall clock at the boundary, an aware value is converted",
        {
            **{ingress: _cases(ingress, *_WINDOW_SPELLING_CONDITIONS) for ingress in (S, P, SB, M)},
            D: (
                *_cases(D, *_WINDOW_SPELLING_CONDITIONS),
                _consumer("experiment_window_compatibility", "window_native_supported"),
                _consumer("experiment_window_compatibility", "window_old_native_refused"),
            ),
            A: (
                *_cases(A, *_WINDOW_SPELLING_CONDITIONS),
                _consumer("experiment_window_compatibility", "window_artifact_supported"),
                _consumer("experiment_window_compatibility", "window_artifact_context_refused"),
            ),
        },
        table="stored-state",
        notes={
            D: "a registration made before the window was bound must be re-registered",
            A: "the context binds the window days; a context without them must be republished",
        },
    ),
    _row(
        "artifact-publication-abort",
        "Publication abort after the manifest insert began invalidates the generation "
        "(tombstone first, then relation erase; if the tombstone cannot be written the "
        "relations are kept and named, and a fresh process hides the generation before "
        "they are dropped)",
        {
            **{
                ingress: (
                    _consumer(
                        "artifact_publication_abort_recovery",
                        f"publication_store_not_comparable_{ingress}",
                    ),
                )
                for ingress in (S, P, SB, M)
            },
            D: (
                _consumer(
                    "artifact_publication_abort_recovery", "publication_from_definitions_supported"
                ),
            ),
            A: (_consumer("artifact_publication_abort_recovery", "dropped_artifact_read_refused"),),
        },
        table="stored-state",
        family=(
            _consumer("artifact_publication_abort_recovery", "abort_tombstones_generation"),
            _consumer("artifact_publication_abort_recovery", "abort_preserves_original_error"),
            _consumer("artifact_publication_abort_recovery", "tombstone_failure_keeps_relations"),
            _consumer(
                "artifact_publication_abort_recovery", "fresh_store_recovers_abandoned_generation"
            ),
        ),
        notes={D: "warehouse artifact store publication"},
    ),
)

ENTRY_ROWS: tuple[EntryRow, ...] = _INGRESS_ROWS + _AUXILIARY_ROWS + _STORED_STATE_ROWS


# --- validation -----------------------------------------------------------------


def validate_catalog(rows: Sequence[EntryRow] = ENTRY_ROWS) -> None:
    """Every row fills its table's columns and every clause names a resolvable
    contract whose evidence is a selector into an existing test module."""
    if set(INGRESSES) != set(CONSTRUCTORS):
        raise ValueError("catalog ingresses must be exactly the Analysis constructors")
    ids: set[str] = set()
    for row in rows:
        if row.id in ids:
            raise ValueError(f"duplicate capability row id {row.id!r}")
        ids.add(row.id)
        if row.table not in TABLE_COLUMNS:
            raise ValueError(f"{row.id}: unknown table {row.table!r}")
        columns = TABLE_COLUMNS[row.table]
        if set(row.bindings) != set(columns):
            raise ValueError(f"{row.id}: bindings must cover exactly the {row.table!r} columns")
        if set(row.notes) - set(columns):
            raise ValueError(f"{row.id}: notes refer to an unknown table column")
        for column, clauses in row.bindings.items():
            where = f"{row.id}/{column}"
            if not clauses:
                raise ValueError(f"{where}: empty cell")
            for clause in clauses:
                _validate_context_text(clause.condition, f"{where} condition")
                _validate_binding_column(where, column, clause.binding)
            resolve_cell(clauses)
            _validate_context_text(row.notes.get(column, ""), f"{where} note")
        if row.family_clauses:
            for clause in row.family_clauses:
                _validate_context_text(clause.condition, f"{row.id}/family condition")
                _validate_family_binding(f"{row.id}/family", clause.binding)
            resolve_cell(row.family_clauses)


def _validate_family_binding(where: str, binding: Binding) -> None:
    """A family clause is a consumer expectation on the family's own ingress,
    never on an Analysis constructor or the auxiliary other-entry-points column."""
    if not isinstance(binding, ConsumerBinding):
        raise ValueError(f"{where}: family clauses must name consumer evidence")
    expectation = _consumer_expectation(binding)
    if expectation.ingress in CONSTRUCTORS or expectation.ingress == OTHER:
        raise ValueError(f"{where}: family evidence is for the {expectation.ingress!r} ingress")
    _validate_evidence(where, expectation.evidence)


def _validate_binding_column(where: str, column: str, binding: Binding) -> None:
    if isinstance(binding, ConsumerBinding):
        expectation = _consumer_expectation(binding)
        allowed = OTHER_INGRESSES + (OTHER,) if column == OTHER else (column,)
        if expectation.ingress not in allowed:
            raise ValueError(f"{where}: consumer evidence is for {expectation.ingress!r}")
        _validate_evidence(where, expectation.evidence)
    elif isinstance(binding, (MatrixBinding, CaseBinding)) and binding.ingress != column:
        raise ValueError(f"{where}: binding is for {binding.ingress!r}")


def _validate_evidence(where: str, evidence: Sequence[str]) -> None:
    if not evidence:
        raise ValueError(f"{where}: consumer evidence reference is empty")
    for reference in evidence:
        _validate_evidence_reference(where, reference)


_SELECTOR = re.compile(
    r"^(?P<module>tests/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)::(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?)$"
)


def _validate_evidence_reference(where: str, reference: str) -> None:
    """A selector names a test module on disk and a test function (or a direct
    class method) defined in it; the module is parsed, never imported or collected."""
    match = _SELECTOR.match(reference)
    if match is None:
        raise ValueError(f"{where}: evidence {reference!r} is not a test selector")
    tree = _module_tree(match["module"])
    if tree is None:
        raise ValueError(f"{where}: evidence module {match['module']} does not exist")
    try:
        resolve_evidence_function(tree, match["name"])
    except ValueError as exc:
        raise ValueError(f"{where}: evidence {reference!r} is unresolved: {exc}") from exc


@cache
def _module_tree(module: str) -> ast.Module | None:
    path = Path(__file__).resolve().parent.parent / module
    if not path.is_file():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


_STATUS_WORDS = re.compile(r"\b(?:supported|refused|warning|not measured|not comparable)\b", re.I)
_CODE_TOKEN = re.compile(r"\b[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+){1,}\b")


def _validate_context_text(text: str, where: str) -> None:
    if _STATUS_WORDS.search(text) or _CODE_TOKEN.search(text) or "`" in text:
        raise ValueError(f"{where} may not introduce an outcome or code")
