"""Verify an interaction scan against a definitions reader's sealed data."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING

from increment._binding import binding_for
from increment._canonical import canonical_json_bytes, declaration_value
from increment._dashboard_capture import DashboardCapture, capture_dashboard
from increment._evidence_loaders import load_interaction_focal_facts, load_interaction_pair_facts
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
        InteractionScan,
        InteractionScanPlan,
        ScanCapture,
    )
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True)
class InteractionDuckDBSource:
    """Read-only definitions reader and original table coordinates for measured pairs."""

    experiment: Experiment
    reader: ExperimentReader
    pair_tables: Mapping[str, Mapping[str, tuple[str, str, str]]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pair_tables",
            MappingProxyType(
                {study: MappingProxyType(dict(feeds)) for study, feeds in self.pair_tables.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class VerifiedInteractionAttachment:
    """Bounded result issued by the qualified source operation; no open reader."""

    value: InteractionAttachment
    manifest: SourceGenerationManifest


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
        or _fixed_utc(definition.start) != scope.start
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


def bind_scan(
    capture: ScanCapture,
    source: InteractionDuckDBSource,
    evaluate: Callable[[ScanCapture], InteractionScan],
) -> VerifiedInteractionAttachment:
    """Issue a result from one verified capture and the source that supplied it."""
    binding = binding_for(source.experiment, source.reader)
    plan = capture.plan
    metrics, recipes = _check_focal_definition(source.experiment, binding, plan)
    scope, focal = plan.scope, plan.scope.focal.entry
    retained = tuple(
        pair.value.evidence for pair in capture.pairs if isinstance(pair.value, CapturedPair)
    )
    if any(item.plan.study_id not in source.pair_tables for item in retained):
        refuse(SCAN_INVALID, field="pair_sources", reason="study_inventory_mismatch")

    def verify(pin: DashboardCapture) -> SourceGenerationManifest:
        pinned = pin.binding.source
        facts = load_interaction_focal_facts(pinned, plan)
        if facts is None:
            refuse(SCAN_INVALID, field="source", reason="focal_facts_not_verified")
        dataset, assignment, outcomes = facts
        feeds = tuple(
            feed
            for item in retained
            for feed in load_interaction_pair_facts(
                pinned, item, source.pair_tables[item.plan.study_id]
            )
        )
        return SourceGenerationManifest(
            source_id=scope.source_id,
            generation_id=f"duckdb-file-sha256:{dataset}",
            dataset_sha256=dataset,
            cutoff=scope.cutoff,
            focal_assignments=assignment,
            focal_outcomes=outcomes,
            pair_feeds=feeds,
        )

    manifest = capture_dashboard(binding, verify, metrics=metrics)
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
        focal_outcome_sha256=manifest.focal_outcomes.content_sha256,
        pair_evidence_sha256=tuple(item.evidence_sha256 for item in retained),
    )
    return VerifiedInteractionAttachment(
        InteractionAttachment(scan=evaluated, focal_receipt=receipt), manifest
    )


def validate_dashboard_interaction(
    attachment: VerifiedInteractionAttachment,
    experiment: Experiment,
    binding: SourceBinding,
    arms: tuple[str, str],
) -> None:
    """Reject mismatched declarations before the dashboard reads any results."""
    if not isinstance(attachment, VerifiedInteractionAttachment):
        refuse(SCAN_INVALID, field="interactions", reason="verified_source_required")
    receipt = attachment.value.focal_receipt
    if (receipt.definition_sha256, receipt.arms) != (experiment.hash(), arms):
        refuse(SCAN_INVALID, field="interactions", reason="different_focal_experiment")
    _, recipes = _check_focal_definition(experiment, binding, attachment.value.scan.plan)
    if recipes != receipt.metrics_sha256:
        refuse(SCAN_INVALID, field="interactions", reason="focal_recipe_mismatch")


def verify_dashboard_interaction(
    pinned: SourceBinding, attachment: VerifiedInteractionAttachment
) -> InteractionScan | None:
    """Independently compare the headline's pinned input with the issued attachment."""
    facts = load_interaction_focal_facts(pinned.source, attachment.value.scan.plan)
    if facts is None:
        return None
    manifest = attachment.manifest
    if facts != (manifest.dataset_sha256, manifest.focal_assignments, manifest.focal_outcomes):
        refuse(SCAN_INVALID, field="interactions", reason="different_source_generation")
    return attachment.value.scan
