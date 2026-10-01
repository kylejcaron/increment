"""Inspect calibration profiles and verify campaign journals."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from calibration.journal import JournalError, verify
from calibration.profile import ProfileError, available, campaigns, load

_I15_MANIFEST = Path(__file__).resolve().parents[1] / "tests" / "_i15_manifest.json"


def _cases(campaign: str) -> list[dict]:
    """The campaign's declared cells, as the records its cell sets select over."""
    if campaign == "i15":
        return json.loads(_I15_MANIFEST.read_text(encoding="utf-8"))["cases"]
    # The sequential ladder admits cells by modelled cost, so its records must
    # carry that cost; the campaign runner owns the declared model.
    from scripts.run_sequential_campaign import manifest_records

    return manifest_records()


def _profiles(campaign: str | None, name: str | None) -> int:
    for campaign_name in [campaign] if campaign else campaigns():
        cases = _cases(campaign_name)
        for profile_name in [name] if name else available(campaign=campaign_name):
            profile = load(profile_name, campaign=campaign_name)
            certification = profile.certification(cases)
            certified = sum(certification.values())
            identifier = profile.cells.id_field
            modelled = sum(
                float(case.get("cost_core_hours", 0.0))
                for case in cases
                if certification[case[identifier]]
            )
            print(
                json.dumps(
                    {
                        "campaign": profile.campaign,
                        "profile": profile.name,
                        "tolerance": profile.tolerance,
                        "delta": profile.delta,
                        "cells": profile.cells.name,
                        "stopping": profile.stopping,
                        "sequential_rule": profile.sequential_rule,
                        "certified_cases": certified,
                        "uncertified_cases": len(certification) - certified,
                        "budget_core_hours": profile.cells.budget,
                        "modelled_core_hours": (
                            round(modelled, 2) if profile.cells.budget is not None else None
                        ),
                    },
                    sort_keys=True,
                )
            )
    return 0


def _verify(directory: Path, case_id: str) -> int:
    totals = verify(directory, case_id=case_id)
    print(
        json.dumps(
            {
                "case_id": totals.case_id,
                "blocks": totals.blocks,
                "started": totals.started,
                "completed": totals.completed,
                "exceptional": totals.exceptional,
                "exceptional_dropped": totals.exceptional_dropped,
                "digest": totals.digest,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibration", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("profiles", help="resolve declared execution profiles")
    listing.add_argument("--campaign", default=None, choices=campaigns())
    listing.add_argument("--name", default=None)
    checking = sub.add_parser("verify", help="verify a case journal's hash chain")
    checking.add_argument("directory", type=Path)
    checking.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "profiles":
            return _profiles(args.campaign, args.name)
        return _verify(args.directory, args.case_id)
    except (ProfileError, JournalError) as error:
        print(f"calibration: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
