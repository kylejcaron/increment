"""Prove the installed WHEEL's public API works with only core dependencies.

Run via `nox -s wheel_smoke`, from a temp directory that is NOT the
repository checkout -- a stray sys.path entry pointing at the source
tree would otherwise let this "pass" against a broken wheel. Exits
nonzero listing every failing name if anything in `increment.__all__`
fails to import, or if `power`'s core-only call chain fails to execute.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import sysconfig

import increment

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _check_duckdb_memtable_from_pyarrow() -> None:
    """Fail if sqlglot breaks Ibis DuckDB create_table from a PyArrow table.

    Needs the demo extra, which the core-only main() forbids, so it runs only under --demo-extras."""
    import ibis
    import pyarrow as pa

    con = ibis.duckdb.connect()
    table = pa.table({"x": [1, 2, 3]})
    created = con.create_table("wheel_smoke_memtable", obj=table, temp=True)
    total = created.x.sum().execute()
    if total != 6:
        raise SystemExit(f"unexpected memtable contents after create_table: {total}")
    con.drop_table("wheel_smoke_memtable")


def main() -> None:
    if any(pathlib.Path(p).resolve() == _REPO_ROOT for p in sys.path if p):
        raise SystemExit(
            "increment resolved with the source checkout on sys.path -- not a wheel-only run"
        )
    if pathlib.Path.cwd().resolve().is_relative_to(_REPO_ROOT):
        raise SystemExit("wheel smoke must run from a directory outside the checkout")
    site_packages = pathlib.Path(sysconfig.get_path("purelib")).resolve()
    if not pathlib.Path(increment.__file__).resolve().is_relative_to(site_packages):
        raise SystemExit(
            f"increment imported from {increment.__file__!r}, not this environment's site-packages"
        )
    optional_modules = ("duckdb", "pandas", "coeftable")
    present = [name for name in optional_modules if importlib.util.find_spec(name) is not None]
    if present:
        raise SystemExit(
            f"core-only wheel environment unexpectedly contains optional modules: {present}"
        )

    failures: list[tuple[str, str]] = []
    for name in increment.__all__:
        try:
            getattr(increment, name)
        except Exception as exc:  # collect every failure, not just the first
            failures.append((name, repr(exc)))
    if failures:
        raise SystemExit(f"root exports failed to import core-only: {failures}")

    from increment.decision import FixedInference
    from increment.estimation.arm_contract import (
        AnalysisAxes,
        ArmPlanningProcedure,
        FamilyPolicy,
        MetricCapabilities,
        PlanningFamilyExpansion,
        RelativeDecisionPolicy,
    )
    from increment.power import Baseline, PowerDesign, achieved_power, required_sample_size
    from increment.semantics.assignment import ParallelAssignment
    from increment.semantics.models import MethodSpec

    procedure = ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=True,
            population="assigned",
            variance_adjustment="none",
        ),
        dependence="iid",
        inference=FixedInference(),
        estimand="mean",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ),
        family_expansion=PlanningFamilyExpansion(family_size=1),
        decision_method=MethodSpec(name="unadjusted", variance_reduction="none"),
        sensitivity_methods=(),
        prior_present=False,
    )
    baseline = Baseline(mean=1.0, var=1.0)
    design = PowerDesign(power=0.8)
    result = achieved_power(300, 0.1, baseline, procedure, design=design)
    assert 0.0 < result.power < design.power, f"unexpected small-sample power: {result.power}"
    sized = required_sample_size(0.1, baseline, procedure, design=design)
    assert sized.n_per_arm > 300, "sizing did not increase the underpowered arm size"
    reached = achieved_power(sized.n_per_arm, 0.1, baseline, procedure, design=design)
    previous = achieved_power(sized.n_per_arm - 1, 0.1, baseline, procedure, design=design)
    assert previous.power < design.power <= reached.power, "sizing failed the target crossing"

    import pyarrow as pa

    from increment import Analysis
    from increment.tables import estimates_to_readout

    data = pa.table(
        {
            "unit": list(range(200)),
            "arm": ["control"] * 100 + ["treatment"] * 100,
            "converted": [0] * 100 + [1] * 40 + [0] * 60,
        }
    )
    estimates = Analysis.from_unit_summary(
        data,
        unit="unit",
        group="arm",
        control="control",
        metrics={"converted": "conversion"},
    ).run()
    (row,) = estimates_to_readout(estimates)
    assert row["lift"] is None and row["lower"] > 0
    assert row["higher"] is None and row["stat_sig"] is True
    assert row["value_scale"] == "relative"
    (frame_row,) = estimates.to_frame(backend="pyarrow").to_pylist()
    assert frame_row["lift"] is None and frame_row["set_lower"] > 0
    assert frame_row["set_upper"] is None

    print(f"wheel_smoke: OK (package={increment.__file__}; cwd={pathlib.Path.cwd()})")


if __name__ == "__main__":
    if "--demo-extras" in sys.argv[1:]:
        _check_duckdb_memtable_from_pyarrow()
        print("wheel_smoke: OK (--demo-extras: sqlglot/DuckDB memtable regression probe passed)")
    else:
        main()
