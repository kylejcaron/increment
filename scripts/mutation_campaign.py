"""Bounded mutmut 3.8 campaigns with a fresh Python interpreter per test run.

Run with the project's dev environment. Generated sources stay in a temporary
copy; JSON verdicts and pytest output are retained in the requested directory.
"""

from __future__ import annotations

import argparse
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
from collections import Counter
from contextlib import chdir
from pathlib import Path

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

    if stage not in {"generate", "stats", "test"}:
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
    else:
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


def campaign(name: str, output: Path) -> None:
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
        for path in ["increment", "tests", *config["also_copy"]]:
            shutil.copytree(ROOT / path, work / path, ignore=shutil.ignore_patterns("__pycache__"))
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
        results = []
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(worker_main(sys.argv[2:]))
    if importlib.metadata.version("mutmut") != "3.8.0":
        raise RuntimeError("This runner requires mutmut 3.8.0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", choices=CAMPAIGNS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in args.campaigns:
        campaign(name, output)


if __name__ == "__main__":
    main()
