from __future__ import annotations

import polars as pl

from increment import AnalysisPlan, from_frame
from tests.analysis_factory import view_readout


def test_source_content_digest_ignores_study_labels():
    data = pl.DataFrame(
        {
            "unit": list(range(8)),
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0],
        }
    )
    digests = []
    for experiment_id in ("warehouse-study", "frame"):
        experiment, resolver = from_frame(
            data,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            plan=AnalysisPlan(primary="value"),
            experiment_id=experiment_id,
        )
        try:
            cells = view_readout(experiment, resolver).as_cells()
            digest = cells[0].identity.source_content_digest
            assert digest is not None
            digests.append(digest)
        finally:
            resolver.close()
    assert digests[0] == digests[1]
