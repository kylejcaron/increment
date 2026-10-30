from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, cast

from pydantic import BaseModel

from increment.declarations import Method
from increment.kernel.contracts import (
    MethodIdentity,
    Scope,
    contract_hash,
    method_config_hash,
    metric_hash,
)
from increment.kernel.contracts import (
    analysis_spec_hash as hash_analysis_spec,
)
from increment.kernel.contracts import (
    execution_identity_hash as hash_execution_identity,
)
from increment.result_envelope import ResultCell
from increment.store.types import Run


@dataclass(frozen=True, slots=True)
class PersistenceIdentity:
    """Effective configuration plus loader-owned source declarations.

    Custom sources without ``declaration_identity()`` use SourceContext alone;
    their cells record why source-declaration identity is unavailable.
    """

    design: BaseModel
    metrics: Mapping[str, BaseModel]
    registration_hash: str
    methods: Mapping[tuple[str, str, str], Method]
    source_refs: Mapping[tuple[str, str], str]
    source_declaration_identity_unavailable_reason: str | None = None


def persist_results(
    results: Any,
    store: Any,
    experiment_id: str,
    run_id: str,
    identity_record: PersistenceIdentity,
) -> None:
    metadata = results.metadata
    if metadata is None or metadata.partial:
        from increment.store._refusals import refuse_store

        refuse_store("store.run.metadata_mismatch", "Only complete run results can be persisted")
    cells = tuple(results.as_cells())
    if not cells:
        from increment.store._refusals import refuse_store

        refuse_store(
            "store.run.empty",
            "An empty metric selection cannot be saved",
            reason="empty_metric_selection",
        )
    try:
        package_version = version("increment")
    except PackageNotFoundError:
        package_version = "0+unknown"

    metric_hashes = {name: metric_hash(metric) for name, metric in identity_record.metrics.items()}
    by_population: dict[Literal["assigned", "triggered"], list[ResultCell]] = {}
    for cell in cells:
        population = cast(Literal["assigned", "triggered"], cell.identity.key.analysis_population)
        by_population.setdefault(population, []).append(cell)
    scopes = {
        population: Scope(view="run", grain="total", population=population)
        for population in by_population
    }
    specifications: dict[str, str] = {}
    method_identities: dict[tuple[str, str, str], MethodIdentity] = {}
    for population, scoped_cells in by_population.items():
        used_methods: dict[str, MethodIdentity] = {}
        source_refs = {
            cell.identity.key.metric: identity_record.source_refs[
                (population, cell.identity.key.metric)
            ]
            for cell in scoped_cells
        }
        for cell in scoped_cells:
            key = cell.identity.key
            estimator = key.method or "unadjusted"
            role = key.method_role or "decision"
            method = identity_record.methods[(key.metric, estimator, role)]
            instance_id = contract_hash(
                "method_instance_v1",
                {"metric": key.metric, "method": estimator, "role": role},
            )
            method_identity = MethodIdentity(
                estimator_id=estimator,
                method_config_hash=method_config_hash(method, method_instance_id=instance_id),
                method_role=role,
                method_instance_id=instance_id,
            )
            used_methods[instance_id] = method_identity
            method_identities[(key.metric, estimator, role)] = method_identity
        metric_names = {cell.identity.key.metric for cell in scoped_cells}
        scoped_metric_hashes = {name: metric_hashes[name] for name in metric_names}
        specifications[population] = hash_analysis_spec(
            design=identity_record.design,
            metric_hashes=scoped_metric_hashes,
            registration_hash=identity_record.registration_hash,
            source_refs=source_refs,
            methods=tuple(used_methods.values()),
            scope=scopes[population],
        )
    analysis_spec_hash = contract_hash(
        "analysis_spec_scopes_v1",
        [[population, specifications[population]] for population in sorted(specifications)],
    )
    execution_identity_hash = hash_execution_identity(
        analysis_spec_hash=analysis_spec_hash,
        implementations={},
        package_version=package_version,
        portable=True,
    )
    identified: list[ResultCell] = []
    for cell in cells:
        key = cell.identity.key
        method = key.method or "unadjusted"
        role = key.method_role or "decision"
        scope = scopes[key.analysis_population]
        cell_spec_hash = specifications[key.analysis_population]
        method_identity = method_identities[(key.metric, method, role)]
        identity = cell.identity.model_copy(
            update={
                "experiment_id": experiment_id,
                "analysis_spec_hash": cell_spec_hash,
                "metric_hash": metric_hashes[key.metric],
                "estimator_id": method_identity.estimator_id,
                "method_config_hash": method_identity.method_config_hash,
                "method_instance_id": method_identity.method_instance_id,
                "scope": scope,
            }
        )
        provenance = replace(
            cell.provenance,
            run_id=run_id,
            execution_identity_hash=execution_identity_hash,
            package_version=package_version,
            source_declaration_identity_unavailable_reason=(
                identity_record.source_declaration_identity_unavailable_reason
            ),
        )
        identified.append(replace(cell, identity=identity, provenance=provenance))
    cells_tuple = tuple(identified)
    now = datetime.now(UTC)
    scope_hashes = tuple(specifications[population] for population in sorted(specifications))
    run = Run(
        experiment_id=experiment_id,
        run_id=run_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        registration_hash=identity_record.registration_hash,
        expected_roster=tuple(cell.identity for cell in cells_tuple),
        cells=cells_tuple,
        metadata=metadata.model_copy(
            update={"run_id": run_id, "analysis_spec_hash": analysis_spec_hash}
        ),
        captured_at=now,
        committed_at=now,
        scope_analysis_hashes=scope_hashes + (analysis_spec_hash,),
        cell_scope_hashes=tuple(
            (cell.identity.hash(), cast(str, cell.identity.analysis_spec_hash))
            for cell in cells_tuple
        ),
    )
    metadata = metadata.model_copy(
        update={"run_id": run_id, "analysis_spec_hash": analysis_spec_hash}
    )
    run = replace(run, metadata=metadata)
    store.put(run)
    results._persisted_cells = cells_tuple
    results._metadata = metadata
