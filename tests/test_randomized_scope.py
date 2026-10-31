from __future__ import annotations

import polars as pl

from increment import Analysis, AnalysisPlan


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
        analysis = Analysis.from_unit_summary(
            data,
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            plan=AnalysisPlan(primary="value"),
            experiment_id=experiment_id,
        )
        try:
            cells = analysis.run().as_cells()
            digest = cells[0].identity.source_content_digest
            assert digest is not None
            digests.append(digest)
        finally:
            analysis.close()
    assert digests[0] == digests[1]
