"""Read one parity view of a builder pair through the public ``run`` route.

A whole-window or breakout read with no population returns every population the
experiment's sources capture, each as its own declared scope of one run; a day-axis
read selects one population and, optionally, one breakout property.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

from increment import Experiment, Scope, run
from increment.evidence import BreakoutIdentity

View = Literal["run", "breakout", "daily", "daily_lift", "asof", "asof_lift"]
Population = Literal["assigned", "triggered"]


@dataclass(frozen=True)
class PopulationReadouts:
    """The readouts of every captured population from one run, iterated as one collection.

    ``metadata`` is the run's metadata, which spans every population's cells.
    """

    parts: tuple[Any, ...]
    metadata: Any

    def __iter__(self) -> Iterator[Any]:
        return chain.from_iterable(self.parts)

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)


def captured_populations(experiment: Experiment) -> tuple[Population, ...]:
    """The populations the experiment's sources capture, assigned first."""
    declared = {
        population
        for reference in experiment.source_refs
        if reference.capabilities is not None
        for population in reference.capabilities.populations
    }
    captured = tuple(
        population for population in ("assigned", "triggered") if population in declared
    )
    return captured or ("assigned",)


def view_scopes(
    experiment: Experiment,
    view: View = "run",
    *,
    population: Population | None = None,
    dimension: str | None = None,
    completed: bool = False,
) -> tuple[Scope, ...]:
    """The scopes one read declares: one per selected population of a whole-window or
    breakout read, or the single scope of a day-axis read."""
    if view in ("run", "breakout"):
        kind = (
            "breakout"
            if view == "breakout"
            else next(
                (scope.view for scope in experiment.scopes if scope.view == "switchback"), "run"
            )
        )
        populations = captured_populations(experiment) if population is None else (population,)
        return tuple(Scope(view=kind, grain="total", population=item) for item in populations)
    axis, _, lift = view.partition("_")
    grain: Literal["daily", "asof"] = "daily" if axis == "daily" else "asof"
    return (
        Scope(
            view=grain,
            grain=grain,
            values=not lift,
            population=population or "assigned",
            segment=(BreakoutIdentity(dimension, None),) if dimension else (),
            completion_policy=completed,
        ),
    )


def collect(result: Any, scopes: tuple[Scope, ...]) -> Any:
    """The read's collection: one scope's readout unchanged, or every population's together."""
    if len(scopes) == 1:
        return result.readout(scopes[0])
    return PopulationReadouts(tuple(result.readout(scope) for scope in scopes), result.metadata)


def read(
    bound: tuple[Experiment, Any],
    view: View = "run",
    *,
    population: Population | None = None,
    dimension: str | None = None,
    completed: bool = False,
    **options: Any,
) -> Any:
    """Run one view of a builder pair and return its readout collection."""
    experiment, resolver = bound
    scopes = view_scopes(
        experiment, view, population=population, dimension=dimension, completed=completed
    )
    return collect(run(experiment.with_views(*scopes), resolver, **options), scopes)
