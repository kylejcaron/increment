"""Run affected tests through a controlled, parser-validated pytest config."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest

_ALLOWED_OPTIONS = {
    "plugins",
    "inifilename",
    "rootdir",
    "tach_base",
    "file_or_dir",
    "markexpr",
    "keyword",
    "maxfail",
    "quiet",
    "verbose",
    "durations",
    "numprocesses",
    "dist",
    "evidence_root",
    "runtime_diagnostics",
    "basetemp",
    "xmlpath",
}


class _AffectedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(
            f"pytest: error: affected runner accepts only typed options ({message})",
            file=sys.stderr,
        )
        raise SystemExit(2)


class _Pytest91Adapter:
    """Isolate pytest 9.1 private parser/config cleanup behind the exact pin."""

    @staticmethod
    def _option_defaults(config: pytest.Config) -> argparse.Namespace:
        return config._parser.parse_known_args([], namespace=argparse.Namespace())

    @classmethod
    def _validate_options(cls, config: pytest.Config) -> None:
        defaults = vars(cls._option_defaults(config))
        actual = vars(config.option)
        unsafe = [
            name
            for name, value in actual.items()
            if name not in _ALLOWED_OPTIONS and value != defaults.get(name)
        ]
        if unsafe:
            raise pytest.UsageError(
                "affected runner accepts only typed options; effective pytest options are "
                + ", ".join(sorted(unsafe))
            )

    @classmethod
    def run(cls, arguments: Sequence[str], plugins: Sequence[str]) -> int:
        from _pytest.config import _prepareconfig

        config = _prepareconfig(list(arguments), plugins=plugins)
        try:
            cls._validate_options(config)
            result = config.hook.pytest_cmdline_main(config=config)
            return int(result or 0)
        except pytest.UsageError as error:
            print(f"pytest: error: {error}", file=sys.stderr)
            return 2
        finally:
            config._ensure_unconfigure()


def _fixed_pytest_config(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    policy = project["tool"]["pytest"]["ini_options"]
    markers = "\n".join(f"    {marker}" for marker in policy["markers"])
    filters = "\n".join(f"    {warning}" for warning in policy["filterwarnings"])
    return (
        "[pytest]\n"
        "addopts =\n"
        "testpaths = tests\n"
        f"markers =\n{markers}\n"
        f"filterwarnings =\n{filters}\n"
    )


def _tier_expression(tier: str, additional: str | None) -> str:
    from scripts._test_tier_policy import tier_mark_expression

    return tier_mark_expression(tier, additional)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _AffectedArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tach-base", required=True)
    parser.add_argument("--tier", choices=("fast", "slow"), required=True)
    parser.add_argument("--keyword", "-k")
    parser.add_argument("--markexpr", "-m")
    parser.add_argument("--maxfail", type=int)
    parser.add_argument("-x", dest="maxfail_one", action="store_true")
    parser.add_argument("-q", dest="quiet", action="count", default=0)
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    parser.add_argument("--durations", type=int)
    parser.add_argument("-n", dest="workers", type=int)
    parser.add_argument("--dist", choices=("loadgroup",))
    parser.add_argument("--evidence-root", default=".test-evidence/affected")
    parser.add_argument("--runtime-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if args.maxfail_one and args.maxfail is not None:
        parser.error("-x and --maxfail cannot be combined")
    if args.workers is not None and not 1 <= args.workers <= 8:
        parser.error("-n must be a numeric worker count from 1 through 8")
    if args.dist is not None and args.workers is None:
        parser.error("--dist requires -n")

    root = Path.cwd().resolve()
    policy_root = Path(__file__).resolve().parents[1]
    for name in tuple(os.environ):
        if name.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_")):
            os.environ.pop(name, None)
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["INCREMENT_AFFECTED_WORKER_ALLOWANCE"] = (
        "serial" if args.workers is None else str(args.workers)
    )
    markexpr = _tier_expression(args.tier, args.markexpr)
    plugins = [
        "tests._evidence",
        "scripts.run_test_tier_plugin",
        "tach.pytest_plugin",
        "xdist.plugin",
        "pytest_timeout",
    ]
    pytest_arguments = [
        "-c",
        "",
        "--rootdir",
        str(root),
        "--tach-base",
        args.tach_base,
        "-m",
        markexpr,
        str(root / "tests"),
        "--evidence-root",
        str(Path(args.evidence_root).resolve()),
    ]
    pytest_arguments[:0] = [part for plugin in plugins for part in ("-p", plugin)]
    if args.keyword:
        pytest_arguments.extend(("-k", args.keyword))
    if args.maxfail_one:
        pytest_arguments.append("-x")
    elif args.maxfail is not None:
        pytest_arguments.extend(("--maxfail", str(args.maxfail)))
    pytest_arguments.extend(["-q"] * args.quiet)
    pytest_arguments.extend(["-v"] * args.verbose)
    if args.durations is not None:
        pytest_arguments.extend(("--durations", str(args.durations)))
    if args.workers is not None:
        pytest_arguments.extend(("-n", str(args.workers), "--dist", args.dist or "loadgroup"))
    if args.runtime_diagnostics:
        pytest_arguments.append("--runtime-diagnostics")

    with tempfile.TemporaryDirectory(prefix="increment-affected-pytest-") as temporary:
        fixed_config = Path(temporary) / "pytest.ini"
        fixed_config.write_text(_fixed_pytest_config(policy_root), encoding="utf-8")
        pytest_arguments[pytest_arguments.index("-c") + 1] = str(fixed_config)
        return _Pytest91Adapter.run(pytest_arguments, plugins)


if __name__ == "__main__":
    raise SystemExit(main())
