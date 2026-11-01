"""Resolve built-in source bindings and execute their experiment views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

from increment._binding import (
    ResolverBinding,
    SourceBinding,
    _execution_unavailable,
    _schema_for_source,
    experiment_for_binding,
)
from increment._execute import execute_views, operation_parameters, selected_binding
from increment._runnability import operation_options, scope_refusals
from increment.errors import CodedError

if TYPE_CHECKING:
    from increment.estimation.readout_types import ScopedReadout
    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.experiment import Experiment


def _reopen_artifact(binding: SourceBinding) -> object:
    from increment.query.source import open_artifact

    spec = binding.artifact_open
    assert spec is not None
    return open_artifact(
        spec.store,
        spec.ref,
        expected_context=spec.expected_context,
        verification=spec.verification,
    )


class SourceResolver(ResolverBinding):
    """Own a built-in source binding and execute the experiment declared over it."""

    def __init__(
        self,
        binding: SourceBinding,
        experiment: Experiment,
        *,
        run_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(binding, experiment, reopen_artifact=_reopen_artifact)
        self._run_options = dict(run_options or {})
        self._refuse_store = run_options is None

    def run_experiment(
        self,
        experiment: Experiment,
        provider: object,
        *,
        options: Mapping[str, object] | None = None,
    ) -> tuple[ScopedReadout, ...]:
        """Admit every selected view before any read, then execute them in one capture."""
        if provider is not self._source:
            _execution_unavailable(experiment, "declarations")
        merged = {**self._run_options, **(options or {})}
        for scope in experiment.scopes:
            refusals = scope_refusals(experiment, scope)
            if refusals:
                raise refusals[0]
            selected_binding(self.binding, experiment, scope)
            kwargs = operation_options(experiment, scope, merged)
            if self._refuse_store and kwargs.get("store") is not None:
                raise CodedError(
                    "Experiment execution does not support the Analysis.run store option",
                    code="kernel.run.invalid_input",
                    context={"field": "store"},
                )
        scoped = execute_views(
            self.binding,
            experiment,
            experiment.scopes,
            options=merged,
            execution_context=cast(
                "EvidenceExecutionContext | None", merged.get("_execution_context")
            ),
        )
        if self.catalog_writer is not None:
            schema = _schema_for_source(self._source)
            self.catalog_writer(
                replace(
                    self.experiment,
                    source_refs=tuple(
                        replace(reference, schema=schema)
                        for reference in self.experiment.source_refs
                    ),
                )
            )
        return scoped


def resolver_for(
    binding: SourceBinding,
    *,
    run_options: Mapping[str, object] | None = None,
    record_schema: bool = False,
) -> tuple[Experiment, SourceResolver]:
    """Declare the experiment a binding executes and the resolver that owns it."""
    if run_options is not None:
        run_options = {"population": None, **run_options}
    population = cast(
        Literal["assigned", "triggered"], (run_options or {}).get("population") or "assigned"
    )
    experiment = experiment_for_binding(
        binding,
        operation_parameters=operation_parameters(),
        record_schema=record_schema,
        population=population,
    )
    return experiment, SourceResolver(binding, experiment, run_options=run_options)


__all__ = ["SourceResolver", "resolver_for"]
