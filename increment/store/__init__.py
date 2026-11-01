from increment.store.base import ResultsStore
from increment.store.decision_log import DecisionEvent, DecisionLog, DecisionReference
from increment.store.parquet import ParquetResultsStore
from increment.store.types import HistorySelector, Run, RunHandle
from increment.store.warehouse import WarehouseResultsStore

__all__ = [
    "DecisionEvent",
    "DecisionLog",
    "DecisionReference",
    "HistorySelector",
    "ParquetResultsStore",
    "ResultsStore",
    "Run",
    "RunHandle",
    "WarehouseResultsStore",
]


def core_operations():
    """Return the values and digest operation required by core run handling."""
    from types import SimpleNamespace

    from increment.store.types import run_digest

    return SimpleNamespace(
        DecisionEvent=DecisionEvent,
        DecisionReference=DecisionReference,
        Run=Run,
        run_digest=run_digest,
    )
