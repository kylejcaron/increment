"""Bounded mutmut 3.8 campaigns with a fresh Python interpreter per test run.

Run with the project's dev environment. Generated sources stay in a temporary
copy; JSON verdicts and pytest output are retained in the requested directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping
from contextlib import chdir
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
RUNNER_ERROR = 70
COLLECTION_ERROR = 71
CAMPAIGNS = {
    "tails": (
        "increment/estimation/_tails.py",
        "tests/estimation/test_tails.py",
        ["two_sided_critical_value", "resolvable_expm1"],
    ),
    "variance": (
        "increment/estimation/variance.py",
        "tests/estimation/test_variance.py",
        ["stable_log_ratio"],
    ),
    "armstats": (
        "increment/estimation/armstats.py",
        "tests/estimation/test_armstats.py",
        ["scaled_cross_over_n"],
    ),
    "meta": (
        "increment/estimation/meta.py",
        "tests/estimation/test_meta.py::TestCochranQ",
        ["cochran_q"],
    ),
    "quantile": (
        "increment/estimation/quantile.py",
        "tests/estimation/test_quantile.py",
        ["_log_quantile_se_impl", "quantile_half_width", "_bracket_half_width"],
    ),
    "power": (
        "increment/power/_validation.py",
        "tests/power/test_curve.py::TestSharedValidationHelpers",
        ["_require_finite", "_require_relative_domain"],
    ),
    "sequential_screen_grid": (
        "increment/estimation/_certified.py",
        "tests/estimation/test_sequential_screen_acceptance.py",
        ["log_interval", "log_rising"],
    ),
    "cuped_evidence_grid": (
        "increment/_cuped_evidence.py",
        "tests/test_cuped_evidence.py",
        [
            "_validate_fitted_union",
            "_validate_fitted_tuple",
            "_validate_fitted_float",
            "_validate_fitted_wire_value",
            "_validate_fitted_wire_dataclass",
            "_restore_wire_value",
            "_fitted_dataclass",
            "_decode_fitted_evidence",
            "parse_cuped_evidence",
        ],
    ),
}


def run_tests(tests: list[str], *, forced_fail: bool) -> int:
    import pytest
    from mutmut.mutation.trampoline import set_mutant_under_test
    from mutmut.runners.harness import PytestRunner

    class Outcomes:
        collection_failed = False
        test_failed = False

        def pytest_collectreport(self, report: pytest.CollectReport) -> None:
            self.collection_failed |= report.failed

        def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
            self.test_failed |= report.failed

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_call(self) -> None:
            if forced_fail:
                set_mutant_under_test("fail")

    outcomes = Outcomes()
    runner = PytestRunner()
    with chdir("mutants"):
        # Preserve pytest's result before mutmut can turn usage errors into exceptions.
        code = int(
            pytest.main(
                ["--rootdir=.", "--tb=native"]
                + runner._pytest_args_regular_run(tests)
                + runner._pytest_add_cli_args,
                plugins=[outcomes],
            )
        )
    print(f"PYTEST_EXIT {code}", flush=True)
    if outcomes.collection_failed and code in (1, 2, 4):
        return COLLECTION_ERROR
    if code == 1 and not outcomes.test_failed:
        return RUNNER_ERROR
    return code


def worker(stage: str, source: str, tests: list[str]) -> int:
    # mutmut's internal generation/stats API is deliberately version-pinned.
    from mutmut.__main__ import (
        copy_also_copy_files,
        copy_src_dir,
        create_mutants,
        setup_source_paths,
    )
    from mutmut.mutation.trampoline import set_mutant_under_test
    from mutmut.runners.harness import PytestRunner
    from mutmut.state import state

    if stage not in {"generate", "stats", "test", "matrix"}:
        raise ValueError(f"Unknown worker stage: {stage}")
    if stage == "generate":
        Path("mutants").mkdir()
        copy_src_dir()
        copy_also_copy_files()
        setup_source_paths()
        create_mutants(1)
        return 0
    setup_source_paths()
    forced_fail = os.environ.get("MUTANT_UNDER_TEST") == "fail"
    if forced_fail:
        set_mutant_under_test("")
    module = importlib.import_module(source.removesuffix(".py").replace("/", "."))
    module_file = module.__file__
    if module_file is None:
        raise RuntimeError(f"Module {module.__name__} has no __file__")
    expected = (Path("mutants") / source).resolve()
    if Path(module_file).resolve() != expected:
        raise RuntimeError(f"Wrong source imported: {module_file}; expected {expected}")
    print(f"SOURCE {module_file}", flush=True)
    if stage == "stats":
        set_mutant_under_test("stats")
        code = PytestRunner().run_stats(tests=[])
        Path("associations.json").write_text(
            json.dumps(
                {
                    key: sorted(value)
                    for key, value in state().tests_by_mangled_function_name.items()
                },
                indent=2,
            )
        )
        return code
    if stage == "matrix":
        import pytest

        junit = Path(os.environ["MUTATION_JUNIT"])
        with chdir("mutants"):
            runner = PytestRunner()
            pytest_args = without_exit_first(
                runner._pytest_args_regular_run(tests) + runner._pytest_add_cli_args
            )
            code = int(
                pytest.main(["--rootdir=.", "--tb=native", "--junitxml", str(junit)] + pytest_args)
            )
        return code
    code = run_tests(tests, forced_fail=forced_fail)
    return code


def worker_main(args: list[str]) -> int:
    try:
        if importlib.metadata.version("mutmut") != "3.8.0":
            raise RuntimeError("This runner requires mutmut 3.8.0")
        return worker(args[0], args[1], args[2:])
    except (Exception, SystemExit):
        traceback.print_exc()
        return RUNNER_ERROR


def classify(code: int | None) -> str:
    # A negative code means the worker died by signal (POSIX). That is either
    # the mutant crashing the interpreter or the harness itself dying (OOM
    # kill, segfault on import), so it gets its own status. RUNNER_ERROR is
    # returned explicitly when the harness could not run the mutant at all.
    if code is not None and code < 0:
        return "crashed"
    return {
        0: "survived",
        1: "killed",
        5: "no tests",
        COLLECTION_ERROR: "collection error",
        None: "timeout",
    }.get(code, "runner error")


def _rates(counts: Counter) -> dict[str, float | None]:
    # Crashes can reflect a dead harness rather than a caught mutant, so exclude
    # them from rate numerators while retaining them in counts for inspection.
    killed = counts["killed"]
    settled = killed + counts["survived"]
    classified = sum(counts.values())
    return {
        # Rate among mutants whose tests actually ran to a kill/survive verdict;
        # excludes mutants with no tests, a timeout, a collection error, or a crash.
        "executed_kill_rate": killed / settled if settled else None,
        # Honest headline: killed over every classified mutant, so neither an
        # unexecuted mutant nor a crash can inflate the rate.
        "score": killed / classified if classified else None,
    }


def greedy_cover(kills: Mapping[str, set[str]]) -> tuple[list[str], dict[str, int]]:
    uncovered = set().union(*kills.values()) if kills else set()
    cover: list[str] = []
    while uncovered:
        best = max(kills, key=lambda test: len(kills[test] & uncovered))
        gain = kills[best] & uncovered
        if not gain:
            break
        cover.append(best)
        uncovered -= gain
    return cover, {
        test: len(mutants & set().union(*(kills[name] for name in cover)))
        for test, mutants in kills.items()
        if test not in cover and mutants
    }


def failing_tests(junit: Path) -> set[str]:
    root = ET.parse(junit).getroot()
    failures = set()
    for case in root.iter("testcase"):
        if not any(child.tag in {"failure", "error"} for child in case):
            continue
        file = case.attrib.get("file")
        if file:
            module_path = Path(file)
            module_parts = module_path.with_suffix("").as_posix().split("/")
            classes = case.attrib.get("classname", "").split(".")[len(module_parts) :]
            nodeid = f"{module_path.as_posix()}::{'::'.join(classes + [case.attrib['name']])}"
        else:
            nodeid = (
                f"{case.attrib.get('classname', '').replace('.', '/')}.py::{case.attrib['name']}"
            )
        failures.add(nodeid)
    return failures


def without_exit_first(args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_value = False
    for arg in args:
        if skip_value:
            skip_value = False
            continue
        if arg in {"-x", "--exitfirst", "--maxfail"}:
            skip_value = arg == "--maxfail"
        elif not arg.startswith("--maxfail="):
            filtered.append(arg)
    return filtered


def effective_kills(kills: Mapping[str, set[str]], string_only: set[str]) -> dict[str, set[str]]:
    return {test: mutants - string_only for test, mutants in kills.items()}


def string_literal_mutants(metadata: Mapping[str, Mapping[str, str]]) -> set[str]:
    def normalize(line: str) -> str:
        tree = ast.parse(line)

        class Strings(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    return ast.copy_location(ast.Constant(value="STRING"), node)
                return node

        return ast.dump(Strings().visit(tree))

    output = set()
    for mutant, diff in metadata.get("mutants", {}).items():
        removed = [
            line[1:].strip()
            for line in diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        added = [
            line[1:].strip()
            for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        try:
            if (
                len(removed) == len(added) == 1
                and normalize(removed[0]) == normalize(added[0])
                and removed[0] != added[0]
            ):
                output.add(mutant)
        except SyntaxError:
            continue
    return output


def collect_kill_matrix(
    name: str,
    source: str,
    output: Path,
    work: Path,
    env: dict[str, str],
    log: Path,
    results: list[dict[str, Any]],
    mutant_diffs: dict[str, str],
) -> dict[str, Any]:
    attributed: dict[str, set[str]] = {
        test: set()
        for result in results
        if result["status"] == "killed"
        for test in result["tests"]
    }
    for result in results:
        if result["status"] != "killed" or not result["tests"]:
            continue
        mutant = result["mutant"]
        junit = output / f"{name}-{hashlib.sha256(mutant.encode()).hexdigest()}.xml"
        code, _ = execute(
            "matrix",
            source,
            result["tests"],
            work,
            env | {"MUTANT_UNDER_TEST": mutant, "MUTATION_JUNIT": str(junit)},
            log,
            30,
        )
        if code not in (0, 1):
            raise RuntimeError(f"Kill matrix worker failed for {mutant}; see {log}")
        for test in failing_tests(junit):
            attributed.setdefault(test, set()).add(mutant)
        junit.unlink()
    literals = string_literal_mutants({"mutants": mutant_diffs})
    effective = effective_kills(attributed, literals)
    cover, candidates = greedy_cover(effective)
    return {
        "cover": cover,
        "candidates_for_removal": candidates,
        "killed_nothing": sorted(test for test, mutants in effective.items() if not mutants),
        "string_literal_only": sorted(literals),
    }


def execute(
    stage: str,
    source: str,
    tests: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout: float,
) -> tuple[int | None, float]:
    started = time.monotonic()
    with log.open("a") as stream:
        stream.write(f"\nSTAGE {stage} MUTANT {env.get('MUTANT_UNDER_TEST', '')}\n")
        stream.flush()
        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--worker", stage, source, *tests],
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return result.returncode, time.monotonic() - started
        except subprocess.TimeoutExpired:
            stream.write("WALL TIMEOUT\n")
            return None, time.monotonic() - started


def campaign(name: str, output: Path, *, kill_matrix: bool = False) -> None:
    source, selection, functions = CAMPAIGNS[name]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    env["MUTANT_UNDER_TEST"] = ""
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        env[key] = "1"
    log = output / f"{name}.log"
    if log.exists() or (output / f"{name}.json").exists():
        raise FileExistsError(f"Campaign evidence already exists for {name}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"increment-mutation-{name}-") as directory:
        work = Path(directory)

        config_text = (ROOT / "pyproject.toml").read_text()
        config = tomllib.loads(config_text)["tool"]["mutmut"]
        config.update(only_mutate=[source], pytest_add_cli_args_test_selection=[selection])
        (work / "pyproject.toml").write_text(
            config_text.split("[tool.mutmut]")[0]
            + "[tool.mutmut]\n"
            + "\n".join(f"{key} = {json.dumps(value)}" for key, value in config.items())
        )
        for path in {source.split("/")[0], "tests", *config.get("also_copy", [])}:
            if (ROOT / path).exists():
                shutil.copytree(
                    ROOT / path, work / path, ignore=shutil.ignore_patterns("__pycache__")
                )
        stages = {}
        for stage, active, wanted in [
            ("generate", "", 0),
            ("stats", "", 0),
            ("test", "", 0),
            ("test", "fail", 1),
        ]:
            code, elapsed = execute(
                stage, source, [], work, env | {"MUTANT_UNDER_TEST": active}, log, 120
            )
            stages["forced_fail" if active else stage] = {"exit_code": code, "seconds": elapsed}
            if code != wanted:
                raise RuntimeError(
                    f"{name}: {stage}/{active} returned {code}, expected {wanted}; see {log}"
                )
        associations = json.loads((work / "associations.json").read_text())
        metadata = json.loads((work / "mutants" / f"{source}.meta").read_text())
        module = source.removesuffix(".py").replace("/", ".")
        prefixes = [f"{module}.x_{function}__mutmut_" for function in functions]
        mutants = sorted(
            key
            for key in metadata["exit_code_by_key"]
            if any(key.startswith(prefix) for prefix in prefixes)
        )
        if not mutants or any(
            not any(key.startswith(prefix) for key in mutants) for prefix in prefixes
        ):
            raise RuntimeError("Requested function has no generated mutants")
        mutant_diffs: dict[str, str] = {}
        results: list[dict[str, Any]] = []
        for mutant in mutants:
            tests = associations.get(mutant.split("__mutmut_")[0], [])
            code, elapsed = (
                execute("test", source, tests, work, env | {"MUTANT_UNDER_TEST": mutant}, log, 30)
                if tests
                else (5, 0.0)
            )
            status = classify(code)
            results.append(
                {
                    "mutant": mutant,
                    "status": status,
                    "exit_code": code,
                    "seconds": elapsed,
                    "tests": tests,
                }
            )
            print(f"{name} {len(results)}/{len(mutants)} {status}: {mutant}", flush=True)
            if kill_matrix and status == "killed":
                import subprocess as _subprocess

                diff_result = _subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from mutmut.mutation.diff_apply import get_diff_for_mutant; "
                        "import sys; print(get_diff_for_mutant(sys.argv[1]))",
                        mutant,
                    ],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                mutant_diffs[mutant] = diff_result.stdout
            counts = Counter(result["status"] for result in results)
            report = {
                "schema_version": 3,
                "runner_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                "mutmut": importlib.metadata.version("mutmut"),
                "python": sys.version,
                "source": source,
                "source_sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
                "selection": selection,
                "functions": functions,
                "marker": "not slow and not parameter_recovery",
                "worker": "fresh interpreter; serial; BLAS threads=1",
                "timeout_seconds": 30,
                "stages": stages,
                "generated_in_file": len(metadata["exit_code_by_key"]),
                "selected": len(mutants),
                "unclassified": len(mutants) - len(results),
                "counts": dict(counts),
                **_rates(counts),
                "seconds": time.monotonic() - started,
                "results": results,
            }
            (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
            if status == "runner error":
                raise RuntimeError(f"Runner error for {mutant}; see {log}")
        if kill_matrix:
            report["kill_matrix"] = collect_kill_matrix(
                name, source, output, work, env, log, results, mutant_diffs
            )
            (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(worker_main(sys.argv[2:]))
    if importlib.metadata.version("mutmut") != "3.8.0":
        raise RuntimeError("This runner requires mutmut 3.8.0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", choices=CAMPAIGNS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kill-matrix", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in args.campaigns:
        campaign(name, output, kill_matrix=args.kill_matrix)


if __name__ == "__main__":
    main()
