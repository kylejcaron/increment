"""Verify an interaction scan against a definitions reader's sealed data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING

from increment._binding import binding_for
from increment._canonical import canonical_json_bytes, declaration_value
from increment._dashboard_capture import DashboardCapture, capture_dashboard
from increment._evidence_loaders import (
    load_interaction_combination_cells,
    load_interaction_combination_registration,
    load_interaction_focal_facts,
    load_interaction_pair_facts,
)
from increment._overlap_refusals import SCAN_INVALID
from increment.errors import refuse
from increment.semantics.interaction_scan import (
    CapturedPair,
    FocalBindingReceipt,
    InteractionAttachment,
    ScanWindow,
    SourceGenerationManifest,
)
from increment.semantics.models import ConversionMetric, MeanMetric

if TYPE_CHECKING:
    from increment._binding import ExperimentReader, SourceBinding
    from increment.kernel.experiment import Experiment
    from increment.semantics.interaction_scan import (
        CombinationCell,
        CombinationResult,
        FocalSourceFacts,
        InteractionCombination,
        InteractionScan,
        InteractionScanPlan,
        ScanCapture,
    )
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True)
class InteractionDuckDBSource:
    """Read-only definitions reader and its independently published generation manifest."""

    experiment: Experiment
    reader: ExperimentReader
    manifest: SourceGenerationManifest


@dataclass(frozen=True, slots=True, init=False)
class VerifiedInteractionAttachment:
    """Bounded result issued by the qualified source operation; no open reader."""

    value: InteractionAttachment
    facts: FocalSourceFacts
    combinations: tuple[CombinationResult, ...]


def interaction_metric_sha256(metric: Metric) -> str:
    """Identify the exact saved outcome recipe, even when not shown on the dashboard."""
    return sha256(canonical_json_bytes(declaration_value(metric))).hexdigest()


def _fixed_utc(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)


def _check_focal_definition(
    experiment: Experiment, binding: SourceBinding, plan: InteractionScanPlan
) -> tuple[tuple[Metric, ...], str]:
    scope = plan.scope
    focal = scope.focal.entry
    definition = binding.experiment
    if definition is None or binding.definitions is None:
        refuse(SCAN_INVALID, field="source", reason="sealed_duckdb_required")
    if (
        experiment.experiment_id != focal.identity.experiment_id
        or definition.unit != focal.unit_namespace
        or definition.control_group != focal.arms[0]
        or (definition.allocation and set(definition.allocation) != set(focal.arms))
    ):
        refuse(SCAN_INVALID, field="focal", reason="definition_identity_mismatch")
    horizon = definition.observation_horizon
    if (
        definition.cluster is not None
        or definition.design is not None
        or definition.day_boundary != "UTC"
        or (
            definition.plan.inference is not None
            and definition.plan.inference.sequential is not None
        )
        or datetime.combine(definition.start_day, time.min, UTC) != scope.start
        or horizon is None
        or datetime.combine(_fixed_utc(horizon).date() + timedelta(days=1), time.min, UTC)
        != scope.end
    ):
        refuse(SCAN_INVALID, field="focal", reason="incompatible_fixed_window")
    by_name = {metric.name: metric for metric in binding.definitions.metrics}
    matching = []
    recipes = []
    for planned in plan.metrics:
        metric = by_name.get(planned.name)
        supported = (isinstance(metric, ConversionMetric) and planned.kind == "binary_mean") or (
            isinstance(metric, MeanMetric)
            and metric.winsorization is None
            and planned.kind == "mean"
        )
        if (
            not supported
            or metric is None
            or metric.entity != definition.unit
            or interaction_metric_sha256(metric) != planned.definition_sha256
        ):
            refuse(SCAN_INVALID, field="metrics", reason="focal_recipe_mismatch")
        matching.append(metric)
        recipes.append((planned.name, planned.definition_sha256))
    return tuple(matching), sha256(canonical_json_bytes(recipes)).hexdigest()


def _check_manifest_focal(manifest: SourceGenerationManifest, facts: FocalSourceFacts) -> None:
    if (facts.generation_id, facts.dataset_sha256) != (
        manifest.generation_id,
        manifest.dataset_sha256,
    ):
        refuse(SCAN_INVALID, field="source", reason="different_source_generation")
    if (facts.assignments, facts.outcomes) != (manifest.focal_assignments, manifest.focal_outcomes):
        refuse(SCAN_INVALID, field="source", reason="focal_facts_not_verified")
    if facts.exposures != (manifest.focal_exposures or manifest.focal_assignments):
        refuse(SCAN_INVALID, field="source", reason="focal_exposures_not_verified")


def bind_scan(
    capture: ScanCapture,
    source: InteractionDuckDBSource,
    evaluate: Callable[[ScanCapture], InteractionScan],
    evaluate_joint: Callable[
        [ScanCapture, dict[InteractionCombination, tuple[CombinationCell, ...]], bool],
        tuple[CombinationResult, ...],
    ],
) -> VerifiedInteractionAttachment:
    """Issue a result from one verified capture and the source that supplied it."""
    binding = binding_for(source.experiment, source.reader)
    plan = capture.plan
    metrics, recipes = _check_focal_definition(source.experiment, binding, plan)
    scope, focal = plan.scope, plan.scope.focal.entry
    retained = tuple(
        pair.value.evidence for pair in capture.pairs if isinstance(pair.value, CapturedPair)
    )
    manifest = source.manifest
    scope.require_cutoff(manifest.cutoff)
    if (manifest.source_id, manifest.scan_plan_id, manifest.definition_sha256) != (
        scope.source_id,
        plan.plan_id,
        source.experiment.hash(),
    ):
        refuse(SCAN_INVALID, field="manifest", reason="focal_definition_mismatch")
    published = {}
    for feed in manifest.pair_feeds:
        published.setdefault(feed.study_id, {})[feed.role] = feed.fact
    if any(item.plan.study_id not in published for item in retained):
        refuse(SCAN_INVALID, field="manifest", reason="study_inventory_mismatch")

    declarations = {item.candidates: item for item in plan.combinations}
    if any(feed.candidates not in declarations for feed in manifest.combination_feeds):
        refuse(SCAN_INVALID, field="combination_feeds", reason="unknown_combination")

    def verify(
        pin: DashboardCapture,
    ) -> tuple[FocalSourceFacts, dict[InteractionCombination, tuple[CombinationCell, ...]], bool]:
        pinned = pin.binding.source
        facts = load_interaction_focal_facts(pinned, plan)
        if facts is None:
            refuse(SCAN_INVALID, field="source", reason="focal_facts_not_verified")
        _check_manifest_focal(manifest, facts)
        for item in retained:
            declared = published[item.plan.study_id]
            tables = {role: fact.coordinate for role, fact in declared.items()}
            proven = load_interaction_pair_facts(pinned, item, tables)
            if declared != {feed.role: feed.fact for feed in proven}:
                refuse(SCAN_INVALID, field="manifest", reason="pair_evidence_mismatch")
        catalog = facts.assignments.coordinate[0]
        captured = {
            combination: load_interaction_combination_cells(
                pinned, catalog, plan, combination, feed
            )
            for feed in manifest.combination_feeds
            for combination in (declarations[feed.candidates],)
        }
        registration = plan.combination_registration
        registered = registration is not None and load_interaction_combination_registration(
            pinned, catalog, registration
        )
        return facts, captured, registered

    facts, joint, verified_registration = capture_dashboard(
        binding, verify, metrics=metrics, interaction_facts=True
    )
    evaluated = evaluate(capture)
    receipt = FocalBindingReceipt(
        scan_plan_id=plan.plan_id,
        capture_id=evaluated.capture_id,
        definition_sha256=source.experiment.hash(),
        metrics_sha256=recipes,
        identity=focal.identity,
        arms=(focal.arms[0], focal.arms[1]),
        unit_namespace=focal.unit_namespace,
        population="assigned",
        window=ScanWindow(
            entry_open=scope.entry_open, start=scope.start, end=scope.end, cutoff=scope.cutoff
        ),
        source_id=manifest.source_id,
        generation_id=manifest.generation_id,
        dataset_sha256=manifest.dataset_sha256,
        focal_assignment_sha256=manifest.focal_assignments.content_sha256,
        focal_outcome_sha256=sha256(
            canonical_json_bytes(
                [(key, fact.content_sha256) for key, fact in manifest.focal_outcomes]
            )
        ).hexdigest(),
        pair_evidence_sha256=tuple(item.evidence_sha256 for item in retained),
    )
    attachment = object.__new__(VerifiedInteractionAttachment)
    object.__setattr__(
        attachment, "value", InteractionAttachment(scan=evaluated, focal_receipt=receipt)
    )
    object.__setattr__(attachment, "facts", facts)
    object.__setattr__(
        attachment, "combinations", evaluate_joint(capture, joint, verified_registration)
    )
    return attachment


def interaction_pin_metrics(
    attachment: VerifiedInteractionAttachment | None,
    experiment: Experiment,
    binding: SourceBinding,
    arms: tuple[str, str],
    display_metrics: tuple[Metric, ...],
) -> tuple[Metric, ...]:
    """Include hidden scan recipes in the pin without exposing them on the dashboard."""
    if attachment is None:
        return display_metrics
    if not isinstance(attachment, VerifiedInteractionAttachment):
        refuse(SCAN_INVALID, field="interactions", reason="verified_source_required")
    receipt = attachment.value.focal_receipt
    if (receipt.definition_sha256, receipt.arms) != (experiment.hash(), arms):
        refuse(SCAN_INVALID, field="interactions", reason="different_focal_experiment")
    metrics, recipes = _check_focal_definition(experiment, binding, attachment.value.scan.plan)
    if recipes != receipt.metrics_sha256:
        refuse(SCAN_INVALID, field="interactions", reason="focal_recipe_mismatch")
    return tuple({metric.name: metric for metric in (*display_metrics, *metrics)}.values())


def verify_dashboard_interaction(
    pinned: SourceBinding, attachment: VerifiedInteractionAttachment
) -> InteractionScan | None:
    """Independently compare the headline's pinned input with the issued attachment."""
    facts = load_interaction_focal_facts(pinned.source, attachment.value.scan.plan)
    if facts is None:
        return None
    if facts.horizons != attachment.facts.horizons:
        refuse(SCAN_INVALID, field="interactions", reason="different_focal_observation_horizon")
    if facts != attachment.facts:
        refuse(SCAN_INVALID, field="interactions", reason="different_source_generation")
    return attachment.value.scan
