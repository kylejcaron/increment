"""Typed loader for calibration execution profiles.

A profile selects which declared cells execute, the enumerated tolerance they
run under, and the stopping rule. It never supplies a free tolerance and never
overrides the hashed manifest.

Stopping is declared in two pieces. ``stopping`` names the CLASS of rule --
``fixed`` or ``sequential`` -- because that is what the certificate's claim
depends on and it stays true across procedure changes. ``sequential_rule``
names the exact versioned procedure, so replacing one sequential procedure
with another bumps a rule id that is pinned in the evidence instead of
quietly redefining what ``sequential`` meant. The valid rule ids are the ones
:mod:`calibration.stopping` implements, so a profile can never advertise a
procedure nothing can run.

Two campaigns declare profiles in the same document. Inference calibration
owns the document root; sequential inference certification owns the nested
``sequential`` section. Both draw from the same ``tolerances`` and
``stopping`` enumerations at the root, because those are what a certificate's
claim is stated in and one campaign must not be able to invent a tolerance the
other has never heard of. What differs is only the cell sets, the profiles,
and the shape of the manifest records a cell set selects over, which is why a
campaign is a declared object here rather than a string a caller passes around.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml

from calibration.stopping import SEQUENTIAL_RULES, FixedDesign, StoppingRule, build_rule

PROFILES_PATH = Path(__file__).with_name("profiles.yaml")

Stopping = Literal["fixed", "sequential"]
_CELL_RULES = frozenset({"every-case", "one-per-axis-combination", "cheapest-within-budget"})
# A declared tolerance is stated at this nominal error and scales down
# proportionally below it. At the strict tolerance that is exactly
# ``tests.mc.scientific_delta``: ``min(.005, .1*q) == .005 * min(1, q/.05)``.
_REFERENCE_NOMINAL_ERROR = 0.05


class ProfileError(ValueError):
    """A profile declaration is unknown, malformed, or not one of the enumerations."""


@dataclass(frozen=True, slots=True)
class Campaign:
    """Where one campaign's profiles live and how its manifest names a cell.

    ``section`` is the document key holding the campaign's ``cell_sets`` and
    ``profiles``; ``None`` means the document root. ``axis_field`` is the
    sub-mapping an axis is read from, or ``None`` when the manifest record is
    flat.
    """

    name: str
    section: str | None
    id_field: str
    axis_field: str | None


CAMPAIGNS: Mapping[str, Campaign] = MappingProxyType(
    {
        "i15": Campaign("i15", None, "case_id", "dgp"),
        "sequential": Campaign("sequential", "sequential", "name", None),
    }
)
DEFAULT_CAMPAIGN = "i15"


def campaigns() -> tuple[str, ...]:
    """Declared campaign names, sorted."""
    return tuple(sorted(CAMPAIGNS))


@dataclass(frozen=True, slots=True)
class CellSet:
    """A declared, deterministic rule for selecting cells from the manifest."""

    name: str
    rule: str
    axes: tuple[str, ...]
    id_field: str = "case_id"
    axis_field: str | None = "dgp"
    budget: float | None = None
    cost_field: str = "cost_core_hours"

    def _axis(self, case: dict[str, Any], axis: str) -> Any:
        if self.axis_field is None:
            return case.get(axis)
        return case.get(self.axis_field, {}).get(axis)

    def _within_budget(self, cases: list[dict[str, Any]]) -> tuple[str, ...]:
        """The cheapest cells whose modelled cost together fits the budget.

        Cost, not position in the manifest, decides what a bounded tier
        reaches: the campaign's cost is concentrated in a few cells, so index
        order would spend a small budget on whatever happened to be declared
        first. Ties break on manifest order so the selection is a function of
        the manifest alone.
        """
        missing = [
            case[self.id_field]
            for case in cases
            if not isinstance(case.get(self.cost_field), (int, float))
        ]
        if missing:
            raise ProfileError(
                f"cell set {self.name!r} selects on {self.cost_field!r}, which "
                f"{len(missing)} cases do not carry (first: {missing[0]!r})"
            )
        if self.budget is None:
            raise ProfileError(f"cell set {self.name!r} declares no budget")
        spent, chosen = 0.0, set()
        for cost, _, identifier in sorted(
            (float(case[self.cost_field]), order, case[self.id_field])
            for order, case in enumerate(cases)
        ):
            if spent + cost > self.budget:
                break
            spent += cost
            chosen.add(identifier)
        return tuple(case[self.id_field] for case in cases if case[self.id_field] in chosen)

    def select(self, cases: list[dict[str, Any]]) -> tuple[str, ...]:
        """Case ids this set certifies, in manifest order."""
        if self.rule == "every-case":
            return tuple(case[self.id_field] for case in cases)
        if self.rule == "cheapest-within-budget":
            return self._within_budget(cases)
        seen: set[tuple[Any, ...]] = set()
        chosen: list[str] = []
        for case in cases:
            key = tuple(self._axis(case, axis) for axis in self.axes)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(case[self.id_field])
        return tuple(chosen)


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """One executable profile; `delta` always comes from a named tolerance."""

    name: str
    campaign: str
    tolerance: str
    delta: float
    # The same tolerance as an exact rational. A campaign that decides its
    # gates in exact arithmetic must not inherit a binary-float tolerance
    # through the door, and a YAML decimal literal has an exact rational
    # value that ``float`` alone discards.
    tolerance_fraction: Fraction
    cells: CellSet
    stopping: Stopping
    sequential_rule: str | None

    def certification(self, cases: list[dict[str, Any]]) -> dict[str, bool]:
        """Per-case certified flag, so omitted cells are reported, not dropped."""
        selected = set(self.cells.select(cases))
        identifier = self.cells.id_field
        return {case[identifier]: case[identifier] in selected for case in cases}

    def tolerance_at(self, nominal_error: float) -> float:
        """The excess-error tolerance this profile allows at *nominal_error*.

        The declared tolerance is stated at ``_REFERENCE_NOMINAL_ERROR`` and
        scales down proportionally below it: a flat allowance would swamp a
        small nominal error. Above the reference the declared value stands.
        """
        if not 0.0 < nominal_error < 1.0:
            raise ProfileError(f"nominal_error must be in (0, 1), got {nominal_error}")
        return self.delta * min(1.0, nominal_error / _REFERENCE_NOMINAL_ERROR)

    def stopping_rule(self, *, nominal_error: float, eta: float, repetitions: int) -> StoppingRule:
        """The executable stopping rule this profile declares, for one cell."""
        design = FixedDesign.resolve(
            nominal_error=nominal_error,
            tolerance=self.tolerance_at(nominal_error),
            eta=eta,
            repetitions=repetitions,
        )
        return build_rule(design, stopping=self.stopping, sequential_rule=self.sequential_rule)


def _document(path: Path | None = None) -> dict[str, Any]:
    text = (path or PROFILES_PATH).read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ProfileError("profiles document must be a mapping")
    return document


def _section(document: dict[str, Any], campaign: str) -> tuple[Campaign, dict[str, Any]]:
    declared = CAMPAIGNS.get(campaign)
    if declared is None:
        raise ProfileError(f"unknown campaign {campaign!r}; declared: {campaigns()}")
    if declared.section is None:
        return declared, document
    nested = document.get(declared.section)
    if not isinstance(nested, dict):
        raise ProfileError(f"profiles document declares no {campaign!r} campaign section")
    return declared, nested


def available(path: Path | None = None, *, campaign: str = DEFAULT_CAMPAIGN) -> tuple[str, ...]:
    """Declared profile names for *campaign*, sorted."""
    _, section = _section(_document(path), campaign)
    profiles = section.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ProfileError(f"profiles document declares no {campaign!r} profiles")
    return tuple(sorted(profiles))


def load(
    name: str, path: Path | None = None, *, campaign: str = DEFAULT_CAMPAIGN
) -> CalibrationProfile:
    """Resolve *name* against the declared enumerations, or raise ``ProfileError``."""
    document = _document(path)
    declared, section = _section(document, campaign)
    profiles = section.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ProfileError(
            f"unknown {campaign} profile {name!r}; declared: {available(path, campaign=campaign)}"
        )
    entry = profiles[name]
    required = {"tolerance", "cells", "stopping"}
    allowed = required | {"sequential_rule"}
    if not isinstance(entry, dict) or not required <= set(entry) <= allowed:
        raise ProfileError(
            f"profile {name!r} must declare {sorted(required)} and at most {sorted(allowed)}"
        )

    tolerances = document.get("tolerances")
    if not isinstance(tolerances, dict) or entry["tolerance"] not in tolerances:
        raise ProfileError(f"profile {name!r} names an undeclared tolerance")
    delta = tolerances[entry["tolerance"]]
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not 0 < delta < 1:
        raise ProfileError(f"tolerance {entry['tolerance']!r} must be a fraction in (0, 1)")

    stopping = document.get("stopping")
    if not isinstance(stopping, list) or entry["stopping"] not in stopping:
        raise ProfileError(f"profile {name!r} names an undeclared stopping rule")
    sequential_rule = entry.get("sequential_rule")
    if entry["stopping"] == "sequential":
        if sequential_rule not in SEQUENTIAL_RULES:
            raise ProfileError(
                f"profile {name!r} stops sequentially but names "
                f"{sequential_rule!r}; implemented: {sorted(SEQUENTIAL_RULES)}"
            )
    elif sequential_rule is not None:
        raise ProfileError(
            f"profile {name!r} stops {entry['stopping']!r} and must not name a sequential rule"
        )

    sets = section.get("cell_sets")
    if not isinstance(sets, dict) or entry["cells"] not in sets:
        raise ProfileError(f"profile {name!r} names an undeclared cell set")
    cell_set = sets[entry["cells"]]
    if not isinstance(cell_set, dict) or cell_set.get("rule") not in _CELL_RULES:
        raise ProfileError(f"cell set {entry['cells']!r} must name a supported rule")
    axes = cell_set.get("axes", [])
    if not isinstance(axes, list) or not all(isinstance(axis, str) for axis in axes):
        raise ProfileError(f"cell set {entry['cells']!r} axes must be strings")
    if cell_set["rule"] == "one-per-axis-combination" and not axes:
        raise ProfileError(f"cell set {entry['cells']!r} requires at least one axis")
    budget = cell_set.get("budget_core_hours")
    if cell_set["rule"] == "cheapest-within-budget":
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
            raise ProfileError(f"cell set {entry['cells']!r} requires a positive budget_core_hours")
    elif budget is not None:
        raise ProfileError(
            f"cell set {entry['cells']!r} does not select on cost and must not declare a budget"
        )

    return CalibrationProfile(
        name=name,
        campaign=campaign,
        tolerance=entry["tolerance"],
        delta=float(delta),
        tolerance_fraction=Fraction(str(delta)),
        cells=CellSet(
            name=entry["cells"],
            rule=cell_set["rule"],
            axes=tuple(axes),
            id_field=declared.id_field,
            axis_field=declared.axis_field,
            budget=None if budget is None else float(budget),
        ),
        stopping=entry["stopping"],
        sequential_rule=sequential_rule,
    )
