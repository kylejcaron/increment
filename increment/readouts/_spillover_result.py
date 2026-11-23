"""Assemble the registered policy comparisons from reduced group evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING

from increment._canonical import canonical_digest_bytes
from increment._spillover_design import SPILLOVER_ESTIMATOR
from increment._spillover_registration import require_registered_spillover
from increment.declarations import CellKey
from increment.estimation.readout_types import (
    CellRecord,
    FamilyScope,
    ReadoutMetadata,
    ReadoutScope,
    SamplingInference,
    SourceReadoutScope,
    SpilloverFamilyTerms,
    cell_order,
    family_identity,
)
from increment.estimation.spillover import SpilloverInterval
from increment.estimation.spillover_results import SpilloverEffect
from increment.kernel.contracts import CellIdentity, contract_hash, metric_hash
from increment.kernel.ports import program_store_operations
from increment.result_envelope import ResultCell
from increment.semantics.models import ConversionMetric

if TYPE_CHECKING:
    from increment._literals import Backend
    from increment._run import RunResult
    from increment._spillover_design import PolicyCell, SpilloverPopulation
    from increment.kernel.experiment import Experiment
    from increment.kernel.ports import ResultsStorePort


_FAMILY = "Registered policy comparisons"


def _capture_id(registration_hash: str, cells: Sequence[PolicyCell]) -> str:
    # These are reduced policy totals, not a fingerprint of the underlying input rows.
    summaries = sorted(
        (c.stratum, c.policy, c.groups, c.mean_total, c.ss, c.unoffered, c.events) for c in cells
    )
    payload = ("spillover.policy_summaries.v1", registration_hash, summaries)
    return "sha256:" + sha256(canonical_digest_bytes(payload)).hexdigest()


def _metadata(
    rows: Sequence[SpilloverEffect],
    terms: SpilloverFamilyTerms,
    *,
    run_id: str,
    analysis_spec_hash: str,
) -> tuple[ReadoutMetadata, FamilyScope]:
    first = rows[0]
    snapshot = first.source_snapshot_id
    records = [
        CellRecord(
            cell=CellKey.from_row(row),
            source_snapshot_id=snapshot,
            sampling=SamplingInference(
                available=row.sampling_available, reason_code=row.unavailable_reason
            ),
        )
        for row in rows
    ]
    registered = tuple(record.cell for record in records)
    records.sort(key=lambda record: cell_order(record.cell))
    keys = tuple(record.cell for record in records)
    family = FamilyScope(
        family_id=first.family_id,
        analysis_population="assigned",
        source_snapshot_id=snapshot,
        view="spillover",
        dimension=None,
        source=None,
        name=_FAMILY,
        identity_metric=first.metric,
        policy_interval=terms,
        members=registered,
        complete=True,
    )
    source = SourceReadoutScope(
        source_snapshot_id=snapshot,
        cells=keys,
        decision_cells=(),
        rosters=(),
        decision_complete_by_population={},
    )
    metadata = ReadoutMetadata(
        scope=ReadoutScope(
            snapshot_id=snapshot,
            cells=keys,
            decision_cells=(),
            populations=("assigned",),
            families=(family,),
            by_source={snapshot: source},
        ),
        cells=tuple(records),
        run_id=run_id,
        analysis_spec_hash=analysis_spec_hash,
        registration_hash=first.registration_hash,
        source_content_digest_unavailable_reason="only_policy_summaries_captured",
    ).with_execution_context(
        snapshot_id=snapshot,
        coherence="captured",
        effective_modes={"policy_summaries": "captured"},
        shared_capture=True,
    )
    return metadata, family


def finish_spillover(
    experiment: Experiment,
    population: SpilloverPopulation,
    cells: Sequence[PolicyCell],
    *,
    result_type: type[RunResult],
    run_id: str,
    results_store: ResultsStorePort | None,
    input_backend: Backend | None = None,
) -> RunResult:
    """Estimate and retain the family, preserving each comparison's declared order."""
    captured = datetime.now(UTC)
    design = population.design
    registration = experiment.registration.registration_hash
    metric = ConversionMetric.model_validate(experiment.metrics[design.outcome])
    assert metric.window_days is not None
    snapshot = _capture_id(registration, cells)
    terms = SpilloverFamilyTerms(window_days=metric.window_days, alpha=design.alpha)
    family_id = family_identity(
        snapshot,
        "assigned",
        "spillover",
        None,
        None,
        _FAMILY,
        identity_metric=metric.name,
        policy_interval=terms,
    )
    method = require_registered_spillover(experiment)
    rule = SpilloverInterval(population, method=method)
    policy_count = len(design.compared_policies)
    rows = []
    for comparison in design.comparisons:
        point, lower, upper, basis = rule.observed(comparison, cells)
        rows.append(
            SpilloverEffect(
                policy=comparison.policy,
                reference=comparison.reference,
                metric=metric.name,
                window_days=metric.window_days,
                method=method,
                estimate=point,
                lower=lower,
                upper=upper,
                range_basis=basis,
                alpha=design.alpha,
                family_size=len(design.comparisons),
                policy_count=policy_count,
                family_id=family_id,
                registration_hash=registration,
                source_snapshot_id=snapshot,
            )
        )
    spec = experiment.analysis_spec_hash
    execution = experiment.execution_identity_hash
    metadata, family = _metadata(rows, terms, run_id=run_id, analysis_spec_hash=spec)
    law_hash = contract_hash(
        "spillover_method_config", {"method": rows[0].method, "registration": registration}
    )
    metric_id = metric_hash(metric)
    instance = contract_hash("spillover_method_instance", (metric_id, law_hash))
    (scope,) = experiment.views
    results = []
    for row in rows:
        identity = CellIdentity(
            key=CellKey.from_row(row),
            source_snapshot_id=snapshot,
            experiment_id=experiment.experiment_id,
            analysis_spec_hash=spec,
            metric_hash=metric_id,
            estimator_id=SPILLOVER_ESTIMATOR,
            method_config_hash=law_hash,
            method_instance_id=instance,
            scope=scope,
        )
        cell = ResultCell.from_report_payload(
            row_kind="spillover",
            identity=identity,
            label=f"{row.policy!r} minus {row.reference!r}",
            row=row,
            family=family,
            created_at=captured,
        )
        provenance = replace(cell.provenance, run_id=run_id, execution_identity_hash=execution)
        results.append(replace(cell, provenance=provenance))
    record = program_store_operations().Run(
        experiment_id=experiment.experiment_id,
        run_id=run_id,
        analysis_spec_hash=spec,
        execution_identity_hash=execution,
        registration_hash=registration,
        expected_roster=tuple(cell.identity for cell in results),
        cells=tuple(results),
        metadata=metadata,
        captured_at=captured,
        committed_at=datetime.now(UTC),
        cell_scope_hashes=tuple((cell.identity.hash(), spec) for cell in results),
    )
    return result_type._from_record(
        record,
        registration=experiment.registration,
        results_store=results_store,
        input_backend=input_backend,
    )
