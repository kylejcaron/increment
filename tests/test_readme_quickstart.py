from __future__ import annotations

import contextlib
import io
import re
import warnings
from pathlib import Path

from increment import ExperimentReader, RunResult, from_frame
from increment.kernel.experiment import Experiment
from increment.result_envelope import ResultCell
from increment.results import LiftEstimate
from tests.analysis_factory import view_readout

ROOT = Path(__file__).resolve().parents[1]


def _python_blocks(path: Path, heading: str) -> list[str]:
    section = path.read_text().split(f"\n{heading}\n", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    return re.findall(r"```python\s*\n(.*?)\n```", section, flags=re.DOTALL)


def _execute(code: str, label: str, namespace: dict[str, object]) -> str:
    """Run a documented block in ``namespace`` and return what it printed."""
    printed = io.StringIO()
    with warnings.catch_warnings(record=True), contextlib.redirect_stdout(printed):
        warnings.simplefilter("always")
        exec(compile(code, label, "exec"), namespace)
    return printed.getvalue()


def test_legacy_analysis_run_result_type_is_unchanged() -> None:
    import polars as pl

    from increment import AnalysisPlan, Randomized
    from increment.breakout.estimates import LiftEstimates

    frame = pl.DataFrame(
        {
            "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
        }
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
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

    assert isinstance(view_readout(experiment, resolver), LiftEstimates)


def test_readme_quickstart_executes_recommended_run() -> None:
    readme = (ROOT / "README.md").read_text()
    quickstart = readme.split("## Quickstart", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    match = re.search(r"```python\s*\n(.*?)\n```", quickstart, flags=re.DOTALL)
    assert match is not None

    namespace: dict[str, object] = {}
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        exec(compile(match.group(1), "README.md quickstart", "exec"), namespace)

    assert isinstance(namespace["experiment"], Experiment)
    assert isinstance(namespace["reader"], ExperimentReader)
    result = namespace["result"]
    assert isinstance(result, RunResult)
    assert result.effects == tuple(cell.summary for cell in result.cells)
    assert result.decisions.run_id == result.run.run_id
    assert all(isinstance(cell, ResultCell) for cell in result.cells)
    rows = {row.metric: row for row in result.readout() if isinstance(row, LiftEstimate)}
    # The README says float columns are averaged and boolean columns become conversion rates.
    assert {name: row.metric_type for name, row in rows.items()} == {
        "revenue": "mean",
        "converted": "conversion",
    }
    revenue, converted = rows["revenue"].lift, rows["converted"].lift
    # The README reads revenue as a significant increase and conversion as inconclusive.
    assert revenue is not None and revenue.lb is not None
    assert 0 < revenue.lb < revenue.value
    assert converted is not None and converted.lb is not None and converted.ub is not None
    assert converted.lb < 0 < converted.ub
    from increment import render

    for audience in ("brief", "full"):
        assert render(result, audience) == render(result.envelope, audience)


def test_quickstart_guide_reports_the_missing_guardrail_as_unavailable() -> None:
    (example,) = _python_blocks(
        ROOT / "docs" / "guides" / "quickstart.md", "## Report a guardrail that cannot be checked"
    )

    namespace: dict[str, object] = {}
    rendered = _execute(example, "quickstart.md missing guardrail", namespace)

    result = namespace["result"]
    assert isinstance(result, RunResult)
    from increment.results import LiftEstimates

    readout = result.readout()
    assert isinstance(readout, LiftEstimates)
    rows = {row.metric: row for row in readout}
    assert rows["revenue"].lift is not None
    assert rows["complaints"].lift is None
    assert rows["complaints"].failure_code == "readout.cell.missing_metric_observations"
    complaint_lines = [line for line in rendered.splitlines() if line.startswith("complaints")]
    assert complaint_lines
    assert all("unavailable" in line for line in complaint_lines)
