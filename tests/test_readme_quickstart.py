from __future__ import annotations

import re
import warnings
from pathlib import Path

from increment import RunResult
from increment.kernel.experiment import Experiment
from increment.kernel.resolver import Resolver
from increment.result_envelope import ResultCell

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_analysis_run_result_type_is_unchanged() -> None:
    import polars as pl

    from increment import Analysis, AnalysisPlan, Randomized
    from increment.breakout.estimates import LiftEstimates

    frame = pl.DataFrame(
        {
            "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics=["value"],
        plan=AnalysisPlan(primary="value"),
    )

    assert isinstance(analysis.run(), LiftEstimates)


def test_readme_quickstart_executes_recommended_run() -> None:
    readme = (ROOT / "README.md").read_text()
    quickstart = readme.split("## Quickstart", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    match = re.search(r"```python\s*\n(.*?)\n```", quickstart, flags=re.DOTALL)
    assert match is not None

    namespace: dict[str, object] = {}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        exec(compile(match.group(1), "README.md quickstart", "exec"), namespace)

    assert isinstance(namespace["exp"], Experiment)
    assert isinstance(namespace["resolver"], Resolver)
    result = namespace["result"]
    assert isinstance(result, RunResult)
    assert result.effects == tuple(cell.summary for cell in result.cells)
    assert result.decisions.run_id == result.run.run_id
    assert all(isinstance(cell, ResultCell) for cell in result.cells)
    rendered = namespace["rendered"]
    assert isinstance(rendered, str)
    assert rendered
    assert "complaints" in rendered
    assert "unavailable" in rendered
    from increment import render
    from increment.result_envelope import AnalysisEnvelope

    envelope = AnalysisEnvelope(result.cells, result.run.metadata)
    for audience in ("brief", "full"):
        assert render(result, audience) == render(envelope, audience)
