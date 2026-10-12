"""Prove the installed WHEEL's public API works with only core dependencies.

Run via `nox -s wheel_smoke`, from a temp directory that is NOT the
repository checkout -- a stray sys.path entry pointing at the source
tree would otherwise let this "pass" against a broken wheel. Exits
nonzero listing every failing name if anything in `__all__` of `increment`,
`increment.results` or `increment.power` fails to import, or if `power`'s
core-only call chain fails to execute.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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


def main(*, core_only: bool = True) -> None:
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
    if core_only:
        optional_modules = ("duckdb", "pandas", "coeftable")
        present = [name for name in optional_modules if importlib.util.find_spec(name) is not None]
        if present:
            raise SystemExit(
                f"core-only wheel environment unexpectedly contains optional modules: {present}"
            )

    failures: list[tuple[str, str]] = []
    promised_tiers = ("increment", "increment.results", "increment.power")
    for module in map(importlib.import_module, promised_tiers):
        for name in module.__all__:
            try:
                getattr(module, name)
            except Exception as exc:  # collect every failure, not just the first
                failures.append((f"{module.__name__}.{name}", repr(exc)))
    if failures:
        raise SystemExit(f"promised-tier exports failed to import core-only: {failures}")

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
    frame = estimates.to_frame(backend="pyarrow")
    assert isinstance(frame, pa.Table)
    (frame_row,) = frame.to_pylist()
    assert frame_row["lift"] is None and frame_row["set_lower"] > 0
    assert frame_row["set_upper"] is None


def _check_readme_mean_example() -> None:
    """Execute the README dataframe mean-analysis example with its documented inputs."""
    from random import Random

    import polars as pl

    from increment import Analysis
    from increment.estimation.results import LiftEstimate

    rng = Random(0)
    df = pl.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(200)],
            "variant": ["control"] * 100 + ["treatment"] * 100,
            "revenue": [rng.gauss(20 + (i >= 100), 5) for i in range(200)],
        }
    )
    results = Analysis.from_unit_summary(
        df,
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    ).run()
    if len(results) != 1:
        raise SystemExit("README mean-analysis example did not produce one result")
    result = results[0]
    if not isinstance(result, LiftEstimate) or result.metric != "revenue" or result.lift is None:
        raise SystemExit("README mean-analysis example did not produce a revenue lift")
    print(f"README mean-analysis smoke: OK (revenue lift={result.lift.value:+.1%})")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--readme-example", action="store_true")
    modes.add_argument("--demo-extras", action="store_true")
    return parser.parse_args()


def _run() -> None:
    args = _arguments()
    main(core_only=not args.demo_extras)
    if args.expected_version is not None:
        actual = importlib.metadata.version("increment")
        if actual != args.expected_version:
            raise SystemExit(
                f"installed increment version {actual!r} does not match expected "
                f"{args.expected_version!r}"
            )
    if args.readme_example:
        _check_readme_mean_example()
        print(f"wheel_smoke: OK (package={increment.__file__}; cwd={pathlib.Path.cwd()})")
    elif args.demo_extras:
        _check_duckdb_memtable_from_pyarrow()
        print("wheel_smoke: OK (--demo-extras: sqlglot/DuckDB memtable regression probe passed)")
    else:
        print(f"wheel_smoke: OK (package={increment.__file__}; cwd={pathlib.Path.cwd()})")


if __name__ == "__main__":
    _run()
