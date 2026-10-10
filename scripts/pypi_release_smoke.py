"""Install and smoke the exact PyPI release outside the source checkout."""

from __future__ import annotations

import argparse
import collections.abc
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_POLL_SECONDS = 10
_DEADLINE_SECONDS = 300
_RETRYABLE_INSTALL_ERRORS = {
    "uv": (
        "because there is no version of",
        "no matching distribution found",
        "could not find a version",
        "from versions: none",
        "not found in the package index",
        "failed to fetch",
        "temporary failure in name resolution",
        "connection refused",
    ),
    "pip": (
        "no matching distribution found",
        "could not find a version",
    ),
}


def _pypi_available(version: str, timeout: float) -> bool:
    url = f"https://pypi.org/pypi/increment/{urllib.parse.quote(version)}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return metadata.get("info", {}).get("version") == version


def _install_from_pypi(version: str, python: Path, timeout: float) -> None:
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--default-index",
            "https://pypi.org/simple",
            "--python",
            str(python),
            f"increment=={version}",
            "polars",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _smoke_installed_package(
    python: Path, script: Path, version: str, cwd: Path, timeout: float
) -> None:
    subprocess.run(
        [str(python), str(script), "--expected-version", version, "--readme-example"],
        cwd=cwd,
        check=True,
        timeout=timeout,
    )


def _retryable_install_failure(error: subprocess.CalledProcessError) -> bool:
    command = error.cmd
    parts = command.split() if isinstance(command, str) else [str(part) for part in command]
    tools = [Path(part).name for part in parts]
    if tools and tools[0] == "uv" and "pip" in tools:
        installer = "uv"
    elif tools and tools[0] in ("pip", "pip3"):
        installer = "pip"
    elif len(tools) >= 3 and tools[1:3] == ["-m", "pip"]:
        installer = "pip"
    else:
        return False
    output = f"{error.stdout or ''}\n{error.stderr or ''}".lower()
    return any(phrase in output for phrase in _RETRYABLE_INSTALL_ERRORS[installer])


def run_post_publish_smoke(
    version: str,
    python: Path,
    smoke_script: Path,
    *,
    cwd: Path,
    index_probe: collections.abc.Callable[[str, float], bool] | None = None,
    installer: collections.abc.Callable[[str, Path, float], None] | None = None,
    smoke_runner: collections.abc.Callable[[Path, Path, str, Path, float], None] | None = None,
    clock: collections.abc.Callable[[], float] | None = None,
    sleeper: collections.abc.Callable[[float], None] | None = None,
    deadline_seconds: float = _DEADLINE_SECONDS,
    poll_seconds: float = _POLL_SECONDS,
) -> None:
    """Wait for and install the exact version; run the smoke once, never retry it."""
    index_probe = index_probe or _pypi_available
    installer = installer or _install_from_pypi
    smoke_runner = smoke_runner or _smoke_installed_package
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    deadline = clock() + deadline_seconds

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise SystemExit(
                f"publication already occurred, but {version} did not become available "
                f"on PyPI within {deadline_seconds:g} seconds"
            )
        if index_probe(version, min(10, remaining)):
            break
        sleeper(min(poll_seconds, remaining))

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise SystemExit(
                f"publication already occurred, but {version} could not be installed "
                f"from PyPI within {deadline_seconds:g} seconds"
            )
        try:
            installer(version, python, remaining)
        except subprocess.CalledProcessError as exc:
            if not _retryable_install_failure(exc):
                raise
            remaining = deadline - clock()
            if remaining <= 0:
                raise SystemExit(
                    f"publication already occurred, but {version} could not be installed "
                    f"from PyPI within {deadline_seconds:g} seconds: {exc.stderr or exc}"
                ) from exc
            print(
                f"PyPI index is not serving {version} yet ({exc.stderr or exc}); retrying",
                flush=True,
            )
            sleeper(min(poll_seconds, remaining))
        else:
            break

    smoke_runner(python, smoke_script, version, cwd, deadline_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="exact published PEP 440 version")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="increment-pypi-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        smoke = Path(__file__).with_name("wheel_smoke.py").resolve()
        run_post_publish_smoke(args.version, python, smoke, cwd=root)
    print(f"PyPI release smoke passed for increment=={args.version}")


if __name__ == "__main__":
    main()
