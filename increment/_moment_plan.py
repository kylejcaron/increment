"""Format-8 moment vocabulary: variables, masks, wire slots and reduction plans.

The one place that says which centered moments the package reduces, how
they are named on the moments wire, and what the ``x`` slot carries. A
leaf: no ibis, narwhals or pydantic, importable by the query builders,
the frame reducers and the estimation seam alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Var = str
Mask = str | None
MomentKind = Literal["n", "ref", "c1", "c2", "count"]
#: One moment: its kind, its variables (none for a count, one for a
#: reference or first moment, two for a second moment) and its mask.
Moment = tuple[MomentKind, tuple[Var, ...], Mask]


@dataclass(frozen=True, slots=True)
class Slot:
    """One format-8 wire column: which moment of which slot variable(s) it carries.

    For a second moment, ``variables`` is the operand order the producers
    multiply residuals in (``cxy`` is ``dx * dy``).
    """

    kind: MomentKind
    variables: tuple[Var, ...]
    mask: Mask = None


#: The uptake mask: a 0/1 variable gating the masked family.
UPTAKE_MASK = "d"

#: Format-8 wire columns in emission order. Slot variables are the wire's
#: own names -- ``y``, ``x`` (whatever the x slot carries, see
#: X_SLOT_ROLES) and ``den`` -- with ``d`` as the uptake mask.
SLOTS: Mapping[str, Slot] = MappingProxyType(
    {
        "ref_y": Slot("ref", ("y",)),
        "cy1": Slot("c1", ("y",)),
        "cy2": Slot("c2", ("y", "y")),
        "ref_x": Slot("ref", ("x",)),
        "cx1": Slot("c1", ("x",)),
        "cx2": Slot("c2", ("x", "x")),
        "cxy": Slot("c2", ("x", "y")),
        "ref_den": Slot("ref", ("den",)),
        "cden1": Slot("c1", ("den",)),
        "cden2": Slot("c2", ("den", "den")),
        "cyden": Slot("c2", ("y", "den")),
        "cxden": Slot("c2", ("x", "den")),
        "sum_d": Slot("count", (), UPTAKE_MASK),
        "cyd": Slot("c1", ("y",), UPTAKE_MASK),
        "cy2d": Slot("c2", ("y", "y"), UPTAKE_MASK),
        "cxd": Slot("c1", ("x",), UPTAKE_MASK),
    }
)

#: The outcome family every row carries; every other slot is optional.
CORE_SLOTS: tuple[str, ...] = ("ref_y", "cy1", "cy2")
OPTIONAL_SLOTS: tuple[str, ...] = tuple(column for column in SLOTS if column not in CORE_SLOTS)

#: What the x slot carries, keyed by the plan variable that lands in it,
#: and the ``x_role`` declaration written alongside.
X_SLOT_ROLES: Mapping[Var, str] = MappingProxyType(
    {"x": "covariate", "size": "cluster_size", "uptake": "uptake_total"}
)
X_ROLE_VARIABLES: Mapping[str, Var] = MappingProxyType(
    {role: variable for variable, role in X_SLOT_ROLES.items()}
)
#: A materialized x slot with no declared role: a clustered row predating
#: the declaration (it carries ``cxden``, which is how the reader knows).
X_UNDECLARED: Var = "x_undeclared"


def slot_variable(variable: Var) -> Var:
    """Wire slot (``y``, ``x`` or ``den``) that a plan variable lands in."""
    return variable if variable in ("y", "den") else "x"


def x_slot_variable(x_role: str | None) -> Var:
    """Plan variable carried by a materialized x slot declaring *x_role*."""
    return X_UNDECLARED if x_role is None else X_ROLE_VARIABLES[x_role]


@dataclass(frozen=True, slots=True)
class Passthrough:
    """A metadata column an aggregate carries through unchanged."""

    column: str
    reduce: Literal["max", "sum"]
    dtype: Literal["float64", "int64"]


#: Winsorization metadata: constant within a group (max) or additive (sum).
WINSOR_PASSTHROUGH: tuple[Passthrough, ...] = (
    Passthrough("winsor_lower_percentile", "max", "float64"),
    Passthrough("winsor_upper_percentile", "max", "float64"),
    Passthrough("winsor_lower_bound", "max", "float64"),
    Passthrough("winsor_upper_bound", "max", "float64"),
    Passthrough("winsor_n", "sum", "int64"),
    Passthrough("winsor_n_lower", "sum", "int64"),
    Passthrough("winsor_n_upper", "sum", "int64"),
)

#: Exact binary totals are integer metadata, not reconstructed centered moments.
_BINARY_PASSTHROUGH = (Passthrough("successes", "sum", "int64"),)

_SLOT_BY_MOMENT: Mapping[Moment, str] = MappingProxyType(
    {("n", (), None): "n", **{(s.kind, s.variables, s.mask): c for c, s in SLOTS.items()}}
)


def _ordered_moments(
    variables: tuple[Var, ...],
    pairs: tuple[tuple[Var, Var], ...],
    masked: Mapping[str, tuple[tuple[Var, ...], ...]],
) -> tuple[tuple[Moment, ...], tuple[Moment, ...]]:
    """(unmasked, masked) moments in emission order.

    Each variable's family is its reference, first and second moment, then
    its cross moments with every EARLIER variable in declaration order;
    that is the column order every producer has emitted since format 2.
    """
    unmasked: list[Moment] = []
    for index, variable in enumerate(variables):
        unmasked += [
            ("ref", (variable,), None),
            ("c1", (variable,), None),
            ("c2", (variable, variable), None),
        ]
        for earlier in variables[:index]:
            unmasked += [("c2", pair, None) for pair in pairs if set(pair) == {earlier, variable}]
    masked_out: list[Moment] = []
    for mask, entries in masked.items():
        masked_out.append(("count", (), mask))
        masked_out += [("c1" if len(entry) == 1 else "c2", entry, mask) for entry in entries]
    return tuple(unmasked), tuple(masked_out)


@dataclass(frozen=True, slots=True)
class MomentPlan:
    """What one centered reduction emits and what it calls each column.

    ``variables`` are emitted in declaration order (reference, first and
    second moment each); ``pairs`` are the cross second moments, each
    ``(a, b)`` in the operand order the residual product is written;
    ``masked`` lists, per mask, the entries emitted under it (one variable
    = first moment, two = second moment) after the mask's count;
    ``names`` gives every emitted moment its output column;
    ``passthrough`` names metadata columns carried through the aggregate.
    """

    variables: tuple[Var, ...]
    pairs: tuple[tuple[Var, Var], ...]
    masked: Mapping[str, tuple[tuple[Var, ...], ...]]
    names: Mapping[Moment, str]
    passthrough: tuple[Passthrough, ...] = ()

    def __post_init__(self) -> None:
        declared = set(self.variables)
        assert len(declared) == len(self.variables), f"duplicate plan variables: {self.variables!r}"
        for a, b in self.pairs:
            assert a != b and {a, b} <= declared, (
                f"cross pair {(a, b)!r} is not two declared variables"
            )
        for mask, entries in self.masked.items():
            assert mask not in declared, f"mask {mask!r} is also a variable"
            for entry in entries:
                assert len(entry) in (1, 2) and set(entry) <= declared, (
                    f"masked entry {entry!r} under {mask!r} is malformed"
                )
        object.__setattr__(self, "masked", MappingProxyType(dict(self.masked)))
        object.__setattr__(self, "names", MappingProxyType(dict(self.names)))
        missing = [moment for moment in self.moments() if moment not in self.names]
        assert not missing, f"plan emits unnamed moments: {missing!r}"

    def unmasked_moments(self) -> tuple[Moment, ...]:
        return _ordered_moments(self.variables, self.pairs, self.masked)[0]

    def masked_moments(self) -> tuple[Moment, ...]:
        return _ordered_moments(self.variables, self.pairs, self.masked)[1]

    def moments(self) -> tuple[Moment, ...]:
        return (("n", (), None), *self.unmasked_moments(), *self.masked_moments())

    def name(self, moment: Moment) -> str:
        """Output column of *moment*; a second moment's operand order is immaterial."""
        kind, variables, mask = moment
        if moment not in self.names and kind == "c2":
            return self.names[(kind, variables[::-1], mask)]
        return self.names[moment]

    def x_variable(self) -> Var | None:
        """The plan variable landing in the x slot, if any (format-8 plans have exactly one)."""
        candidates = [variable for variable in self.variables if slot_variable(variable) == "x"]
        return candidates[0] if candidates else None

    @classmethod
    def format8(
        cls, variables: tuple[Var, ...], *, passthrough: tuple[Passthrough, ...] = ()
    ) -> MomentPlan:
        """A plan emitting the full format-8 row shape, named by SLOTS.

        *variables* must fill the three slots ``y``, ``x`` and ``den`` once
        each (in whatever order the producer emits its families).
        """
        by_slot = {slot_variable(variable): variable for variable in variables}
        assert len(by_slot) == 3 and set(by_slot) == {"y", "x", "den"}, (
            f"a format-8 plan fills y, x and den exactly once; got {variables!r}"
        )
        pairs = tuple(
            (by_slot[slot.variables[0]], by_slot[slot.variables[1]])
            for slot in SLOTS.values()
            if slot.kind == "c2" and slot.mask is None and slot.variables[0] != slot.variables[1]
        )
        masked = {
            UPTAKE_MASK: tuple(
                tuple(by_slot[s] for s in slot.variables)
                for slot in SLOTS.values()
                if slot.mask == UPTAKE_MASK and slot.kind != "count"
            )
        }
        unmasked, masked_moments = _ordered_moments(variables, pairs, masked)
        to_slot = {variable: slot for slot, variable in by_slot.items()}
        names: dict[Moment, str] = {("n", (), None): "n"}
        for kind, moment_variables, mask in (*unmasked, *masked_moments):
            slot_variables = tuple(to_slot[v] for v in moment_variables)
            key = (kind, slot_variables, mask)
            if key not in _SLOT_BY_MOMENT:
                key = (kind, slot_variables[::-1], mask)
            names[(kind, moment_variables, mask)] = _SLOT_BY_MOMENT[key]
        return cls(variables, pairs, masked, names, passthrough)


#: Unit-grain rows: ``group_summary`` / frame totals, winsor metadata carried.
UNIT_GRAIN = MomentPlan.format8(
    ("y", "x", "den"), passthrough=(*WINSOR_PASSTHROUGH, *_BINARY_PASSTHROUGH)
)
#: Day-axis rows (daily, as-of, cohort): the same shape, no winsor passthrough.
DAY_GRAIN = MomentPlan.format8(("y", "x", "den"), passthrough=_BINARY_PASSTHROUGH)
#: Cluster-grain rows: den before the x slot, as the clustered collapse emits.
CLUSTER_SIZE_GRAIN = MomentPlan.format8(("y", "den", "size"), passthrough=WINSOR_PASSTHROUGH)
CLUSTER_UPTAKE_GRAIN = MomentPlan.format8(("y", "den", "uptake"), passthrough=WINSOR_PASSTHROUGH)

#: Design-level compliance at cluster grain: the bivariate moments of
#: cluster uptake totals and sizes, named as ComplianceArm's wire fields.
COMPLIANCE_CLUSTER = MomentPlan(
    variables=("uptake", "size"),
    pairs=(("uptake", "size"),),
    masked={},
    names={
        ("n", (), None): "n_clusters",
        ("ref", ("uptake",), None): "ref_uptake",
        ("c1", ("uptake",), None): "cluster_uptake1",
        ("c2", ("uptake", "uptake"), None): "cluster_uptake2",
        ("ref", ("size",), None): "ref_size",
        ("c1", ("size",), None): "cluster_size1",
        ("c2", ("size", "size"), None): "cluster_size2",
        ("c2", ("uptake", "size"), None): "cluster_cross",
    },
)


def _compliance_from_cluster_row() -> Mapping[str, str]:
    """ComplianceArm field -> ``group_summary(cluster=..., uptake=True)`` column."""
    renamed = {"size": "den"}
    out: dict[str, str] = {}
    for kind, variables, mask in COMPLIANCE_CLUSTER.moments():
        cluster_moment: Moment = (kind, tuple(renamed.get(v, v) for v in variables), mask)
        out[COMPLIANCE_CLUSTER.name((kind, variables, mask))] = CLUSTER_UPTAKE_GRAIN.name(
            cluster_moment
        )
    return MappingProxyType(out)


COMPLIANCE_ARM_FROM_CLUSTER_ROW: Mapping[str, str] = _compliance_from_cluster_row()
