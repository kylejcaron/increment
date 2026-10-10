"""Every retired code is (a) absent from the live registry and (b) maps to
live code(s) or to None. Prevents a merge/deletion
from silently reviving a code under its old name, and prevents
RETIRED_CODES from pointing at a typo."""

from __future__ import annotations

import importlib
import pkgutil

import increment
from increment.errors import RETIRED_CODES, RefusalSpec


def _live_codes() -> set[str]:
    codes: set[str] = set()
    for _finder, name, _ispkg in pkgutil.walk_packages(increment.__path__, prefix="increment."):
        module = importlib.import_module(name)
        for value in vars(module).values():
            if isinstance(value, RefusalSpec):
                codes.add(value.code)
            elif isinstance(value, dict):
                for spec in value.values():
                    if isinstance(spec, RefusalSpec):
                        codes.add(spec.code)
    return codes


def test_retired_codes_are_absent_from_the_live_registry() -> None:
    live = _live_codes()
    still_live = sorted(set(RETIRED_CODES) & live)
    assert still_live == [], f"retired codes still registered: {still_live}"


def test_retired_code_targets_are_live_or_explicitly_none() -> None:
    live = _live_codes()
    missing_targets = sorted(
        f"{old!r} -> {target!r}"
        for old, new in RETIRED_CODES.items()
        for target in ((new,) if isinstance(new, str) else (new or ()))
        if target not in live
    )
    assert missing_targets == [], f"RETIRED_CODES points at a non-live code: {missing_targets}"


def test_definition_codes_are_registered_without_being_raised() -> None:
    # A lazily registered code would be invisible to every registry gate.
    assert "definition.experiment.observation_end_before_end" in _live_codes()


def test_retired_winsor_positivity_code_resolves_to_the_negative_outcome_code() -> None:
    """A consumer branching on the retired positivity refusal finds its replacement."""
    from increment.errors import CodedError
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import RawArm, WinsorRawState

    replacement = RETIRED_CODES["estimation.winsor.pilot_nonpositive_outcome"]
    assert replacement == "estimation.winsor.pilot_negative_outcome"
    assert replacement in _live_codes()
    raw = WinsorRawState(
        metric="revenue",
        study_id="retired",
        missingness="error",
        quantile=0.5,
        arms=(
            RawArm(group_id="C", values=(-1.0, 1.0, 2.0)),
            RawArm(group_id="T", values=(1.0, 2.0, 3.0)),
        ),
    )
    try:
        estimate_winsor_lift(raw, "C", "T")
    except CodedError as refusal:
        assert refusal.code == replacement
    else:
        raise AssertionError("a negative outcome must refuse")
