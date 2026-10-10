"""Install and smoke the exact PyPI release outside the source checkout."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="exact published PEP 440 version")
    args = parser.parse_args()
    deadline = time.monotonic() + _DEADLINE_SECONDS
    metadata_url = f"https://pypi.org/pypi/increment/{urllib.parse.quote(args.version)}/json"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(
                f"publication already occurred, but {args.version} did not become available "
                f"on PyPI within {_DEADLINE_SECONDS} seconds"
            )
        try:
            with urllib.request.urlopen(metadata_url, timeout=min(10, remaining)) as response:
                metadata = json.load(response)
            if time.monotonic() > deadline:
                raise SystemExit(
                    f"publication already occurred, but {args.version} did not become available "
                    f"on PyPI within {_DEADLINE_SECONDS} seconds"
                )
            if metadata.get("info", {}).get("version") != args.version:
                raise SystemExit(f"PyPI returned unexpected version metadata for {args.version}")
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SystemExit(
                    f"publication already occurred, but {args.version} did not become available "
                    f"on PyPI within {_DEADLINE_SECONDS} seconds: {exc}"
                ) from exc
            print(f"PyPI version not available yet ({exc}); retrying", flush=True)
            time.sleep(min(_POLL_SECONDS, remaining))

    with tempfile.TemporaryDirectory(prefix="increment-pypi-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--default-index",
                "https://pypi.org/simple",
                "--python",
                str(python),
                f"increment=={args.version}",
                "polars",
            ],
            check=True,
        )
        smoke = Path(__file__).with_name("wheel_smoke.py").resolve()
        subprocess.run(
            [str(python), str(smoke), "--expected-version", args.version, "--readme-example"],
            cwd=root,
            check=True,
        )
    print(f"PyPI release smoke passed for increment=={args.version}")


if __name__ == "__main__":
    main()
