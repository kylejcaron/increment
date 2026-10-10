"""Smoke tests for the runnable marimo notebooks under `examples/`.

`marimo export html` runs each notebook end to end (same crash coverage
as script mode) and renders real numbers into the HTML that the content
assertions check. Assertions use structural strings (metric name + role,
section heading), never bare words, exact counts, or pinned decimals:
export can duplicate text between the source panel and rendered output
(making substring counts unreliable), and posterior-derived digits drift
on unrelated dependency bumps. `power.py` is the deterministic
closed-form exception, so pinning its figures is safe.

`marimo check` batches all notebooks once per module, preserving a guard for
each file. All tests are `slow`+`examples` (`tables` extra required for most);
run via `make test-examples`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.examples]

_WAREHOUSE_GROUP = pytest.mark.xdist_group("realistic_demo_warehouse")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
_MARIMO_NOTEBOOKS = tuple(
    path.relative_to(EXAMPLES_DIR).as_posix()
    for path in sorted(EXAMPLES_DIR.rglob("*.py"))
    if "app = marimo.App" in path.read_text(encoding="utf-8")
)

# analysis_from_a_warehouse.py writes this relative to cwd=REPO_ROOT.
_MOMENTS_PARQUET = REPO_ROOT / "increment_moments.parquet"
_REALISTIC_DEMO_WAREHOUSE = EXAMPLES_DIR / "realistic_demo" / "warehouse"

_NOTEBOOK_PYTHONWARNINGS = ",".join(
    (
        "error",
        "ignore:fetch_arrow_table() is deprecated:DeprecationWarning:ibis.backends.duckdb",
    )
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = _NOTEBOOK_PYTHONWARNINGS
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _export_html(notebook: str, tmp_path: Path) -> str:
    out = tmp_path / "notebook.html"
    warehouse_backup = None
    creates_warehouse = notebook in ("data_model.py", "analysis_from_a_warehouse.py")
    if creates_warehouse and _REALISTIC_DEMO_WAREHOUSE.exists():
        warehouse_backup = tmp_path / "realistic-demo-warehouse"
        shutil.move(_REALISTIC_DEMO_WAREHOUSE, warehouse_backup)
    try:
        result = _run(
            sys.executable, "-m", "marimo", "export", "html", f"examples/{notebook}", "-o", str(out)
        )
    finally:
        if notebook == "analysis_from_a_warehouse.py":
            _MOMENTS_PARQUET.unlink(missing_ok=True)
        if creates_warehouse:
            try:
                if _REALISTIC_DEMO_WAREHOUSE.exists():
                    shutil.rmtree(_REALISTIC_DEMO_WAREHOUSE)
            finally:
                if warehouse_backup is not None:
                    shutil.move(warehouse_backup, _REALISTIC_DEMO_WAREHOUSE)
    assert result.returncode == 0, result.stderr
    return out.read_text(encoding="utf-8")


def _check(*notebooks: Path) -> dict[Path, list[dict[str, Any]]]:
    result = _run(
        sys.executable, "-m", "marimo", "check", "--format", "json", *(str(p) for p in notebooks)
    )
    assert result.stdout, result.stderr
    report = json.loads(result.stdout)
    failures: dict[Path, list[dict[str, Any]]] = {}
    for issue in report["issues"]:
        if issue.get("severity") == "breaking" or issue["type"] == "error":
            failures.setdefault(Path(issue["filename"]).resolve(), []).append(issue)
    assert result.returncode == bool(failures), result.stdout + result.stderr
    return failures


@pytest.fixture(scope="module")
def checked_notebooks():
    return _check(*(EXAMPLES_DIR / notebook for notebook in _MARIMO_NOTEBOOKS))


def test_notebook_subprocess_fails_on_unhandled_warning():
    result = _run(
        sys.executable,
        "-c",
        "import warnings; warnings.warn('notebook warning policy probe', UserWarning)",
    )
    assert result.returncode != 0
    assert "notebook warning policy probe" in result.stderr


def test_notebook_subprocess_allows_exact_ibis_warning():
    result = _run(
        sys.executable,
        "-c",
        "import ibis; ibis.duckdb.connect().list_tables()",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.xdist_group("marimo_check")
@pytest.mark.parametrize("notebook", _MARIMO_NOTEBOOKS)
def test_marimo_check_reports_no_diagnostics(notebook, checked_notebooks):
    failures = checked_notebooks.get(EXAMPLES_DIR / notebook, [])
    assert not failures, failures


@pytest.mark.parametrize("breaking", [False, True])
def test_batched_notebook_checks_attribute_only_breaking_diagnostics(tmp_path, breaking):
    valid = tmp_path / "valid.py"
    candidate = tmp_path / "candidate.py"
    source = "import marimo\napp = marimo.App()\n"
    valid.write_text(source)
    if breaking:
        source += (
            "@app.cell\ndef _():\n    value = 1\n    return (value,)\n"
            "@app.cell\ndef _():\n    value = 2\n    return (value,)\n"
        )
    candidate.write_text(source)
    failures = _check(valid, candidate)
    assert set(failures) == ({candidate} if breaking else set())


def _rendered_table_title_index(html: str, title: str) -> int:
    """Find *title*'s position where great_tables actually rendered it,
    not one of its other appearances in the export.

    A rendered title can appear escaped up to three times: the source-code
    panel, the intro cell's markdown prose, and the real `<td class="gt_...">`
    heading great_tables emits. Only the last proves the table rendered, so
    require `gt_` (a great_tables CSS class) within a small window before
    the match - markdown/source text never carries that marker.
    """
    needle = f"\\u003E{title}\\u003C"
    start = 0
    while (idx := html.find(needle, start)) != -1:
        if "gt_" in html[max(0, idx - 200) : idx]:
            return idx
        start = idx + 1
    raise AssertionError(f"no gt_-rendered occurrence of {title!r} found in export")


# --- breakout.py: needs tables (coeftable) plus altair (dev group) --------


def test_breakout_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("breakout.py", tmp_path)
    assert "new_onboarding_v2 readout, by country" in html  # readout_table(..., title=...)
    # `\u003E{code}\u003C` matches the JSON-escaped rendered `<td>` cell,
    # not the notebook's own "US/GB/DE/CA" prose (the duplication trap above).
    for country in ("US", "GB", "DE", "CA"):
        assert f"\\u003E{country}\\u003C" in html

    # Keep the country breakout independent of arm-label spellings; the
    # complete analysis output supplies both arms to the renderer.

    # `polyline` only appears in a rendered <svg> sparkline; each table's
    # count is checked in the window before the NEXT table's title.
    tables_in_order = (
        ("new_onboarding_v2 readout, by country", 5),  # segment table + trend
        ("Metric values over time", 6),  # pooled, both arms overlaid
        ("Metric values over time, by segment", 5),  # 4 segments overlaid
        ("As-of metric values over time", 6),  # pooled as-of, both arms
    )
    # Also bounds by the two (unasserted) lift tables' positions, so the
    # last tracked table's window doesn't spill to end of document.
    boundary_titles = (
        *(t for t, _ in tables_in_order),
        "new_onboarding_v2 readout",
        "new_onboarding_v2 readout, as-of trend",
    )
    boundaries = sorted(_rendered_table_title_index(html, t) for t in boundary_titles)
    for title, min_polylines in tables_in_order:
        start = _rendered_table_title_index(html, title)
        end = min((b for b in boundaries if b > start), default=len(html))
        polylines = html[start:end].count("polyline")
        assert polylines >= min_polylines, (
            f"expected >= {min_polylines} sparkline polylines under {title!r} "
            f"(bounded to the next table's start) -- got {polylines}, "
            "suggesting the series rendered blank for most/all rows"
        )


# Mermaid's ER link token: a cardinality pair either side of `--`
# (identifying) or `..` (non-identifying). A single dash renders as a
# parse-error box in the browser, which an export-only check cannot see.
_MERMAID_ER_RELATIONSHIP = re.compile(
    r"^\w+ (?:\|o|\|\||\}o|\}\|)(?:--|\.\.)(?:o\||\|\||o\{|\|\{) \w+ : \S+$"
)


def _mermaid_diagrams(html: str) -> list[str]:
    """Every diagram string mermaid.js will parse in the export.

    The cell source appears in the export too, so read the rendered
    `marimo-mermaid` element's `data-diagram` instead - that is the exact
    text handed to mermaid, entity-escaped by marimo's HTML serializer.
    """
    return [
        raw.replace("\\u0026#92;n", "\n").replace("\\u0026quot;", "").replace("\\u0026#39;", "'")
        for raw in re.findall(r"marimo-mermaid data-diagram='(.*?)'", html)
    ]


# --- ttest_benchmark.py: seeded dataframe simulation, needs coeftable ------


def test_ttest_benchmark_preserves_half_point_effects():
    from examples import ttest_benchmark

    _, namespace = cast(tuple[Any, Any], ttest_benchmark.effect_selection.run())
    selected_true_effect = namespace.get("selected_true_effect")
    format_true_effect = namespace.get("format_true_effect")

    assert selected_true_effect(4.5) == 0.045
    assert format_true_effect(0.045) == "+4.5%"


def test_ttest_benchmark_export_renders_comparison(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("ttest_benchmark.py", tmp_path)
    assert _rendered_table_title_index(html, "Relative lift comparison") >= 0


# --- power.py: closed-form power-analysis math, no extras needed -----------


def test_power_export_renders_expected_figures(tmp_path):
    html = _export_html("power.py", tmp_path)
    # Deterministic closed-form solve (required_sample_size at lift=0.03,
    # mean=12.40, var=430.0, alpha=0.05, power=0.80): pinning is safe.
    # Values reflect Task 1's own-arm H1 variance model (evaluates the
    # treatment arm's variance at its own mean instead of the null's).
    assert "Treatment arm: 48,803 assigned users" in html
    assert "Total planned sample: 97,606 assigned users" in html
    assert "Estimating a Power Curve" in html
    assert "Sequential planning with asymptotic_mean inference" in html


# --- cuped.py: seeded synthetic warehouse (fixed seed=2025), needs tables


def test_cuped_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("cuped.py", tmp_path)
    assert "Unadjusted vs CUPED" in html
    assert "CUPED reduces" in html
    assert "avg_session_duration" in html
    assert "purchase_rate" in html
    assert "d7_retention" in html


# --- analysis_from_a_warehouse.py: realistic warehouse, needs tables -------


@_WAREHOUSE_GROUP
def test_analysis_from_a_warehouse_export_renders_readouts(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("analysis_from_a_warehouse.py", tmp_path)
    assert _rendered_table_title_index(html, "Checkout Redesign: Headline Readout") >= 0
    assert _rendered_table_title_index(html, "Checkout Redesign: Rehydrated Readout") >= 0


@_WAREHOUSE_GROUP
def test_realistic_data_model_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    warehouse_existed = _REALISTIC_DEMO_WAREHOUSE.exists()
    sentinel = _REALISTIC_DEMO_WAREHOUSE / ".acceptance-sentinel"
    if not warehouse_existed:
        _REALISTIC_DEMO_WAREHOUSE.mkdir()
        sentinel.write_text("preserve me", encoding="utf-8")
    try:
        html = _export_html("data_model.py", tmp_path)
        assert _rendered_table_title_index(html, "Checkout Redesign: Headline Readout") >= 0
        assert "Experiment: Checkout Redesign" in html

        diagrams = _mermaid_diagrams(html)
        assert diagrams, "warehouse ERD did not render a marimo-mermaid element"
        for diagram in diagrams:
            relationships = [line.strip() for line in diagram.splitlines() if " : " in line]
            assert relationships, f"no relationship lines in diagram:\n{diagram}"
            for line in relationships:
                assert _MERMAID_ER_RELATIONSHIP.match(line), f"unparseable mermaid link: {line!r}"
        if not warehouse_existed:
            assert sentinel.read_text(encoding="utf-8") == "preserve me"
    finally:
        if not warehouse_existed:
            shutil.rmtree(_REALISTIC_DEMO_WAREHOUSE)


@_WAREHOUSE_GROUP
def test_realistic_data_model_regenerates_stale_generator_manifest(tmp_path):
    pytest.importorskip("coeftable")
    backup = tmp_path / "realistic-demo-warehouse"
    if _REALISTIC_DEMO_WAREHOUSE.exists():
        shutil.move(_REALISTIC_DEMO_WAREHOUSE, backup)

    _REALISTIC_DEMO_WAREHOUSE.mkdir(parents=True)
    (_REALISTIC_DEMO_WAREHOUSE / "manifest.json").write_text(
        json.dumps({"generator_version": 1}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "notebook.html"
    try:
        result = _run(
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            "examples/data_model.py",
            "-o",
            str(out),
        )
        assert result.returncode == 0, result.stderr

        from examples.realistic_demo import generate as demo_generate

        manifest = json.loads(
            (_REALISTIC_DEMO_WAREHOUSE / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["generator_version"] == demo_generate.GENERATOR_VERSION
    finally:
        shutil.rmtree(_REALISTIC_DEMO_WAREHOUSE)
        if backup.exists():
            shutil.move(backup, _REALISTIC_DEMO_WAREHOUSE)


# --- analysis_from_a_dataframe.py: guided dataframe tutorial ---------------


def test_analysis_from_a_dataframe_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("analysis_from_a_dataframe.py", tmp_path)

    for title in (
        "Checkout experiment",
        "CUPED Comparison",
        "Checkout experiment \\u2014 informative prior",
        "Analyzing Panel Data",
        "Windowed Lift",
        "Retention Lift",
    ):
        assert _rendered_table_title_index(html, title) >= 0
    assert "Avg. Order Value" in html
    assert re.search(
        r"run_daily\(\)\\u003C/code\\u003E produced "
        r"\\u003Cstrong\\u003E\d+\\u003C/strong\\u003E per-day metric values",
        html,
    )


# --- observational.py: no warehouse, needs pandas + tables for readout ----


def test_observational_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("observational.py", tmp_path)
    # Export success plus the named cell proves the final comparison rendered.
    assert '"name": "final_comparison"' in html


# --- encouragement.py: no warehouse, needs pandas + tables for readout ----


def test_encouragement_export_renders_expected_figures(tmp_path):
    pytest.importorskip("coeftable")
    html = _export_html("encouragement.py", tmp_path)

    for label in (
        "ITT \\u2014 prompt:",
        "LATE \\u2014 clicking for compliers:",
    ):
        assert re.search(
            rf"\\u003Cstrong\\u003E{re.escape(label)}\\u003C/strong\\u003E "
            r"[+-]\d+\.\d%",
            html,
        )
    assert re.search(
        r"\\u003Cstrong\\u003EUptake \\u2014 clicking:\\u003C/strong\\u003E "
        r"[+-]\d+\.\d percentage points \(\d+\.\d% vs \d+\.\d%\)",
        html,
    )

    assert "Intervals and assumptions" in html
    assert "ITT \\u2014 effect of the prompt" in html
    assert "Uptake \\u2014 relative lift in clicking" in html
    assert "LATE \\u2014 effect of clicking for compliers" in html
    assert "Required assumptions:" in html
    assert "no user would click without the prompt but refuse when prompted" in html

    assert re.search(
        r"Reported: \\u003Cstrong\\u003Ecompliance, itt\\u003C/strong\\u003E",
        html,
    )
    assert "LATE is omitted" in html
    assert "late suppressed" in html
    assert re.search(r"first-stage z=\d+\.\d\d \\u0026lt; \d+", html)

    assert "Monitor LATE over time" in html
    assert "Always valid" in html
    assert "always_valid" in html
    assert "finalized outcome and uptake windows" in html


# --- hte.py: no warehouse, needs tables (pandas + coeftable) ---------------


def test_hte_export_succeeds(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("coeftable")
    _export_html("hte.py", tmp_path)


# --- late_over_time.py: no warehouse, needs pandas + tables for readout ---


def test_late_over_time_export_renders_expected_figures(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("coeftable")
    html = _export_html("late_over_time.py", tmp_path)

    assert "LATE over time" in html
    assert "statistically meaningless" in html
    assert "The as-of LATE trend" in html
    assert "Fixed-horizon LATE" in html
    assert "Always-valid compliance" in html
    assert re.search(r"[+-]\d+\.\d%, [+-]\d+\.\d%", html)
