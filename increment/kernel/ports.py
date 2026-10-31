"""Typed contracts for optional execution implementations."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.metadata import distribution
from typing import Any, Protocol

from increment.estimation.readout_types import ReadoutMetadata
from increment.kernel.contracts import CellIdentity, CompiledPlanBinding
from increment.result_envelope import AnalysisEnvelope, ResultCell


class CateOperations(Protocol):
    """CATE operations supplied by the observational implementation."""

    estimate_cate: Callable[..., Any]
    validate_cate: Callable[..., Any]
    targeting_rule: Callable[..., Any]
    select_targeting_rule: Callable[..., Any]


class ProgramStoreOperations(Protocol):
    """Value constructors and digest operation supplied by program storage."""

    DecisionEvent: type[Any]
    DecisionReference: type[Any]
    PersistenceIdentity: type[Any]
    Run: type[Any]

    def run_digest(self, run: RunRecord) -> str: ...


class RunRecord(Protocol):
    """Run data consumed by core orchestration without store ownership."""

    @property
    def expected_roster(self) -> tuple[CellIdentity, ...]: ...

    @property
    def cells(self) -> tuple[ResultCell, ...]: ...

    @property
    def metadata(self) -> ReadoutMetadata | None: ...

    @property
    def run_id(self) -> str: ...

    @property
    def experiment_id(self) -> str: ...

    @property
    def registration_hash(self) -> str | None: ...

    @property
    def decision_plan_unavailable_reason(self) -> str | None: ...

    @property
    def decision_plan(self) -> CompiledPlanBinding | None: ...

    @property
    def execution_identity_hash(self) -> str: ...

    @property
    def scope_analysis_hashes(self) -> tuple[str, ...]: ...


class RunHandlePort(Protocol):
    """A persisted or in-memory handle to a program run."""

    @property
    def run(self) -> RunRecord: ...

    def envelope(self) -> AnalysisEnvelope: ...


class ResultsStorePort(Protocol):
    """Minimal store operations consumed by core run orchestration."""

    def put(self, run: Any) -> None: ...

    def open_run(
        self, experiment_id: str, analysis_spec_hash: str, run_id: str
    ) -> RunHandlePort: ...


class DecisionLogPort(Protocol):
    """Append-only decision sink consumed by core orchestration."""

    def append(self, event: Any) -> None: ...


@cache
def cate_operations() -> CateOperations:
    """Load the installed CATE implementation only when a route is used."""
    providers = tuple(
        provider
        for provider in distribution("increment").entry_points
        if provider.group == "increment.cate_operations" and provider.name == "core"
    )
    if len(providers) != 1:
        raise RuntimeError("exactly one built-in increment.cate_operations provider is required")
    return providers[0].load()()


@cache
def program_store_operations() -> ProgramStoreOperations:
    """Load program-store values only when core handles a run."""
    providers = tuple(
        provider
        for provider in distribution("increment").entry_points
        if provider.group == "increment.program_store" and provider.name == "core"
    )
    if len(providers) != 1:
        raise RuntimeError("exactly one built-in increment.program_store provider is required")
    return providers[0].load()()
