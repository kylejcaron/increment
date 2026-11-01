from __future__ import annotations

from typing import Protocol

import pyarrow as pa

from increment.store.types import HistorySelector, Run, RunHandle


class ResultsStore(Protocol):
    def put(self, run: Run) -> None: ...
    def open_run(
        self, experiment_id: str, analysis_spec_hash: str, run_id: str | None = None
    ) -> RunHandle: ...
    def query(
        self, experiment_id: str, analysis_spec_hash: str, run_id: str | None = None
    ) -> pa.Table: ...
    def history(self, experiment_id: str, selector: HistorySelector) -> pa.Table: ...
