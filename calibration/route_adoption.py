"""Explicit, audited adoption of saved bound enumerations, and the static audit behind it.

A saved enumeration may stand for a fresh one only where the code that decided each count pair
is the code that decides it now. ``adopt`` establishes that for one source file and the revisions
its campaign ran at, verifies every record against the current runtime, and writes the records
beside a manifest that pins what was audited; ``read_checkpoint`` then accepts them only while
that manifest and the current decision path still agree. Nothing here runs on an ordinary resume.

The audit is a static comparison of a decision path at two revisions of the repository.

An enumeration of the pipeline's rejection probability is a pure function of the code that
decides each count pair, so a saved enumeration may stand for a fresh one only where that code
is identical. The audit takes entry points (module-level functions, constants and classes), walks
every definition each reaches by name, imports and attribute use, and compares the normalised
syntax tree of each definition at the two revisions. Normalisation drops comments, docstrings
and annotations, which cannot change a result; everything else, including every default and
every constant, is compared.

The walk over-approximates: an attribute call whose receiver is not an imported module reaches
every method of that name in the modules already reached, and a class reaches its construction.
A difference outside the reached set cannot change the path; a difference inside it is reported
and passes only as a reviewed difference pinned to the exact digests it was reviewed at.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Text of a repository file at a revision, or ``None`` when it does not exist there.
Sources = Callable[[str], str | None]

Definition = tuple[str, str]
_CONSTRUCTION = ("__init__", "__post_init__", "__new__")


def git_sources(repo: Path, revision: str) -> Sources:
    """The files of ``revision`` in the repository at ``repo``."""
    cache: dict[str, str | None] = {}

    def read(path: str) -> str | None:
        if path not in cache:
            shown = subprocess.run(
                ["git", "show", f"{revision}:{path}"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            cache[path] = shown.stdout if shown.returncode == 0 else None
        return cache[path]

    return read


def tree_sources(root: Path) -> Sources:
    """The files of the working tree at ``root``."""

    def read(path: str) -> str | None:
        file = root / path
        return file.read_text() if file.is_file() else None

    return read


class _Normalise(ast.NodeTransformer):
    """Drops what cannot change a result: docstrings and annotations."""

    @staticmethod
    def _strip(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._strip(node)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._strip(node)
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._strip(node)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if node.value is None:
            return ast.Expr(ast.Constant(f"field:{ast.unparse(node.target)}"))
        return ast.Assign(targets=[node.target], value=node.value, lineno=0)


def normalised(node: ast.AST) -> ast.AST:
    """A copy of ``node`` without what cannot change a result."""
    return _Normalise().visit(ast.parse(ast.unparse(node)))


def digest(node: ast.AST) -> str:
    """A digest of the normalised syntax tree of ``node``."""
    dumped = ast.dump(normalised(node), include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


def _imports(nodes: Iterable[ast.AST], package: str) -> dict[str, tuple[str, str | None]]:
    """``{local name: (module, member)}`` of the runtime imports among ``nodes`` (a member of
    ``None`` binds the module)."""
    table: dict[str, tuple[str, str | None]] = {}
    for top in nodes:
        for node in ast.walk(top):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    table[local] = (alias.name if alias.asname else local, None)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    parts = parts[: len(parts) - node.level + 1]
                    base = ".".join([*parts, *([base] if base else [])])
                for alias in node.names:
                    table[alias.asname or alias.name] = (base, alias.name)
    return table


def _is_type_checking(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, (ast.Name, ast.Attribute))
        and ast.unparse(node.test).endswith("TYPE_CHECKING")
    )


class _Module:
    """The definitions of one module: functions, constants, classes (a shell without its
    methods) and ``Class.method``."""

    def __init__(self, name: str, text: str, *, is_package: bool) -> None:
        self.name = name
        self.package = name if is_package else name.rpartition(".")[0]
        self.defs: dict[str, ast.AST] = {}
        body = ast.parse(text).body
        self.imports = _imports(
            [node for node in body if not _is_type_checking(node)], self.package
        )
        self._collect(body)

    def _collect(self, body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[node.name] = node
            elif isinstance(node, ast.ClassDef):
                kept = [
                    m
                    for m in node.body
                    if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                shell = ast.ClassDef(
                    node.name,
                    node.bases,
                    node.keywords,
                    kept or [ast.Pass()],
                    node.decorator_list,
                    [],
                )
                self.defs[node.name] = shell
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.defs[f"{node.name}.{member.name}"] = member
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.defs[name.id] = node
            elif isinstance(node, (ast.If, ast.Try)) and not _is_type_checking(node):
                self._collect([*node.body, *getattr(node, "orelse", [])])


class Tree:
    """The Python modules of one revision, parsed on demand."""

    def __init__(self, sources: Sources) -> None:
        self._sources = sources
        self._modules: dict[str, _Module | None] = {}

    def module(self, name: str) -> _Module | None:
        if name not in self._modules:
            stem = name.replace(".", "/")
            module = None
            for path, is_package in ((f"{stem}.py", False), (f"{stem}/__init__.py", True)):
                text = self._sources(path)
                if text is not None:
                    module = _Module(name, text, is_package=is_package)
                    break
            self._modules[name] = module
        return self._modules[name]

    def definition(self, module: str, name: str) -> ast.AST | None:
        found = self.module(module)
        return None if found is None else found.defs.get(name)


def _references(node: ast.AST, module: _Module, tree: Tree) -> tuple[set[Definition], set[str]]:
    """The definitions ``node`` names and the attribute names it uses on other receivers."""
    imports = {**module.imports, **_imports([node], module.package)}
    found: set[Definition] = set()
    attributes: set[str] = set()

    def resolve(local: str) -> tuple[str, str | None] | None:
        if local in imports:
            return imports[local]
        return None

    def include(owner: str, name: str) -> None:
        if (target := tree.module(owner)) is None:
            return
        if name in target.defs:
            found.add((owner, name))
            if isinstance(target.defs[name], ast.ClassDef):
                found.update(
                    (owner, f"{name}.{m}") for m in _CONSTRUCTION if f"{name}.{m}" in target.defs
                )

    def dotted(chain: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(chain, ast.Attribute):
            parts.append(chain.attr)
            chain = chain.value
        if isinstance(chain, ast.Name):
            return ".".join([chain.id, *reversed(parts)])
        return None

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id in module.defs:
                include(module.name, child.id)
            elif (bound := resolve(child.id)) is not None and bound[1] is not None:
                include(bound[0], bound[1])
        elif isinstance(child, ast.Attribute):
            path = dotted(child)
            head, _, rest = (path or "").partition(".")
            bound = resolve(head) if path else None
            if bound is not None and rest:
                owner = bound[0] if bound[1] is None else f"{bound[0]}.{bound[1]}"
                if tree.module(owner) is not None:
                    include(owner, rest.split(".")[0])
                    continue
            attributes.add(child.attr)
    return found, attributes


def closure(
    tree: Tree, entries: Iterable[Definition], *, leaves: Iterable[Definition] = ()
) -> dict[Definition, str]:
    """``{definition: digest}`` of every definition reached from ``entries``. An attribute use
    reaches the method of that name of a reached class when the using definition is a method of
    that class, names it, or lives in its module: the places an instance can come from without
    the use naming its type. A definition among ``leaves`` is compared but not walked into."""
    reached: dict[Definition, str] = {}
    found_by: dict[Definition, set[Definition]] = {}
    uses: dict[Definition, set[str]] = {}
    pending = list(entries)
    stops = set(leaves)
    while pending:
        owner, name = pending.pop()
        if (owner, name) in reached:
            continue
        node = tree.definition(owner, name)
        if node is None:
            raise LookupError(f"{owner}.{name} is not defined")
        reached[owner, name] = digest(node)
        if (owner, name) in stops:
            continue
        module = tree.module(owner)
        assert module is not None
        found, attributes = _references(normalised(node), module, tree)
        found_by[owner, name] = found
        uses[owner, name] = attributes
        pending.extend(found)
        for user, attributes_used in uses.items():
            user_owner, user_name = user
            user_class = user_name.rpartition(".")[0]
            for class_owner, class_name in [
                d for d in reached if isinstance(tree.definition(*d), ast.ClassDef)
            ]:
                local = (
                    user_owner == class_owner
                    or (class_owner, class_name) in found_by[user]
                    or (user_class == class_name and user_owner == class_owner)
                )
                if not local:
                    continue
                members = tree.module(class_owner)
                assert members is not None
                pending.extend(
                    (class_owner, key)
                    for key in members.defs
                    if key.rpartition(".")[0] == class_name
                    and key.rpartition(".")[2] in attributes_used
                )
    return reached


@dataclass(frozen=True, slots=True)
class Difference:
    """A definition that is not the same at both revisions: its digests, ``None`` where it is
    not reached or not defined."""

    definition: Definition
    source: str | None
    destination: str | None


def fingerprint(reached: Mapping[Definition, str]) -> str:
    """A digest of a closure: every definition reached and its digest."""
    lines = "\n".join(
        f"{module}:{name}:{value}" for (module, name), value in sorted(reached.items())
    )
    return hashlib.sha256(lines.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Review:
    """A difference read and found not to change a decision, pinned to the digests it was read
    at: the definition's digest at each audited revision it was reviewed against (``None``
    where it was absent) and the one it was reviewed against in the current tree."""

    sources: frozenset[str | None]
    destination: str | None
    why: str


@dataclass(frozen=True, slots=True)
class Audit:
    """The comparison of one decision path at two revisions."""

    source: Mapping[Definition, str]
    destination: Mapping[Definition, str]
    differences: tuple[Difference, ...] = field(default=())

    def unreviewed(self, reviewed: Mapping[Definition, Review]) -> tuple[Difference, ...]:
        """The differences not covered by a review pinned to exactly their digests."""
        return tuple(
            d
            for d in self.differences
            if (found := reviewed.get(d.definition)) is None
            or d.source not in found.sources
            or d.destination != found.destination
        )


def audit(
    source: Sources,
    destination: Sources,
    entries: Iterable[Definition],
    *,
    source_entries: Iterable[Definition] | None = None,
    leaves: Iterable[Definition] = (),
) -> Audit:
    """Compare the code ``entries`` reach at ``source`` with what they reach at
    ``destination`` (``source_entries`` when the source names them differently), not walking
    into ``leaves``."""
    entries, leaves = tuple(entries), tuple(leaves)
    then = closure(
        Tree(source), source_entries if source_entries is not None else entries, leaves=leaves
    )
    now = closure(Tree(destination), entries, leaves=leaves)
    different = tuple(
        Difference(key, then.get(key), now.get(key))
        for key in sorted(then.keys() | now.keys())
        if then.get(key) != now.get(key)
    )
    return Audit(then, now, different)


def _target(node: ast.AST) -> str:
    """A target as written, without the parentheses ``ast.unparse`` puts round a tuple."""
    text = ast.unparse(node)
    return text[1:-1] if isinstance(node, ast.Tuple) and text.startswith("(") else text


def statements(tree: Tree, module: str, functions: Iterable[str]) -> dict[str, frozenset[str]]:
    """The expressions assigned to each target in ``functions``: ``{target: {digest of value}}``
    for plain and augmented assignments (``x += `` is its own key) and for ``for`` loops
    (``for target``, the digest of what they iterate)."""
    found: dict[str, set[str]] = {}
    for name in functions:
        node = tree.definition(module, name)
        if node is None:
            raise LookupError(f"{module}.{name} is not defined")
        for child in ast.walk(normalised(node)):
            if isinstance(child, ast.Assign):
                keys = [_target(target) for target in child.targets]
                value: ast.AST = child.value
            elif isinstance(child, ast.AugAssign):
                keys = [f"{_target(child.target)} +="]
                value = child.value
            elif isinstance(child, ast.For):
                keys = [f"for {_target(child.target)}"]
                value = child.iter
            else:
                continue
            for key in keys:
                found.setdefault(key, set()).add(digest(ast.Expr(value)))
    return {key: frozenset(values) for key, values in found.items()}


# --- What an enumeration's count-pair decisions are made of --------------------------------

#: The code that decides a count pair's finite-sample rejection, routes it, and builds its
#: windows, as named at the audited revisions and as named in the current tree.
SOURCE_ENTRIES: tuple[Definition, ...] = (
    ("increment.power._binomial", "RejectionGeometry"),
    ("increment.power._binomial", "RejectionGeometry.cells"),
    ("increment.power._binomial", "classify"),
    ("increment.power._binomial", "_runtime_rejects"),
    ("increment.power._binomial", "_window"),
    ("increment.power.core", "_binomial_key"),
    ("increment.estimation.arm_contract", "ArmPlanningProcedure"),
    ("increment.estimation.arm_contract", "ArmPlanningProcedure.standard"),
    ("increment.estimation.arm_contract", "ArmPlanningProcedure.compiled_tail_alpha"),
    ("increment.estimation.conversion_route", "dense_min_count"),
    ("calibration.conversion_route", "_finite_blocks"),
)
DESTINATION_ENTRIES = SOURCE_ENTRIES
#: What an adopted record also stands on in the current tree and the audited revisions did not
#: have: the delta-method decision and the sums, recomputed at adoption and equal to the saved
#: figures then. Any change to them voids the adoption, though no audited revision is compared.
FINGERPRINT_ENTRIES: tuple[Definition, ...] = (
    *DESTINATION_ENTRIES,
    ("calibration.conversion_route", "_design_lattice"),
    ("calibration.conversion_route", "_delta_rejects"),
    ("calibration.conversion_route", "delta_sums"),
    ("calibration.conversion_route", "enumerate_design"),
)

#: The runtime's own row, which the delta decision defers to: named by the runtime construction
#: an adoption is made for, not walked into.
FINGERPRINT_LEAVES: tuple[Definition, ...] = (
    ("increment.estimation.conversion_delta", "production_decision"),
)
#: The delta-method side of the current tree's geometry, which the audited revisions did not have
#: and a finite-sample region request never enters: compared as a definition, not walked into.
AUDIT_LEAVES: tuple[Definition, ...] = (
    *FINGERPRINT_LEAVES,
    ("increment.estimation.conversion_delta", "delta_decision"),
)


#: The statements of the enumeration that fix the finite-sample part and the routing, which the
#: audited revisions wrote in ``enumerated_power`` and the current tree writes in these
#: functions; each must hold the same expressions in both.
SOURCE_ARITHMETIC = ("calibration.conversion_route", ("enumerated_power",))
DESTINATION_ARITHMETIC = (
    "calibration.conversion_route",
    ("_design_lattice", "_decided_blocks", "enumerate_design"),
)
ARITHMETIC_KEYS = (
    "procedure",
    "tail",
    "p_t",
    "key",
    "window_c, window_t",
    "x_t",
    "floor",
    "for rows, plus, minus",
    "x_c",
    "weight",
    "smallest",
    "routed",
    "share",
    "finite_part",
    "share +=",
    "finite_part +=",
)

#: Differences between an audited revision and the current tree that were read and found not
#: to change a decision, pinned to the digests they were read at: a further change to either
#: side voids the review.
REVIEWED: dict[Definition, Review] = {}

LOCK_FILE = "uv.lock"


def route_law() -> str:
    """The current routing law, as ``dense_min_count`` computes it."""
    from increment.estimation import conversion_route as route

    return f"max({route._DENSE_FLOOR}, ceil({route._DENSE_SLOPE} z^4))"


class AdoptionError(ValueError):
    """A saved enumeration that cannot be adopted, with every reason found."""


@dataclass(frozen=True, slots=True)
class RevisionAudit:
    """One audited revision against the current tree."""

    revision: str
    audit: Audit
    unreviewed: tuple[Difference, ...]
    arithmetic: tuple[str, ...]
    lock_unchanged: bool

    @property
    def holds(self) -> bool:
        return not self.unreviewed and not self.arithmetic and self.lock_unchanged

    def problems(self) -> list[str]:
        found = [
            f"{self.revision[:10]}: {d.definition[0]}.{d.definition[1]} is "
            f"{d.source or 'absent'} there and {d.destination or 'absent'} now, not reviewed"
            for d in self.unreviewed
        ]
        found += [
            f"{self.revision[:10]}: the enumeration's {key!r} statements differ"
            for key in self.arithmetic
        ]
        if not self.lock_unchanged:
            found.append(f"{self.revision[:10]}: {LOCK_FILE} differs from the current tree")
        return found


def report(revisions: Sequence[str]) -> int:
    """Print each revision's comparison with the current tree, and exit nonzero unless every
    one holds."""
    repo = repository()
    failed = 0
    for revision in revisions:
        found = audit_revision(repo, revision)
        print(
            f"{found.revision[:10]}: {len(found.audit.source)} definitions reached there, "
            f"{len(found.audit.destination)} now, {len(found.audit.differences)} differ, "
            f"{len(found.unreviewed)} not reviewed; lock "
            f"{'unchanged' if found.lock_unchanged else 'DIFFERS'}"
        )
        for difference in found.audit.differences:
            module, name = difference.definition
            pinned = REVIEWED.get(difference.definition)
            status = (
                "reviewed"
                if pinned
                and difference.source in pinned.sources
                and difference.destination == pinned.destination
                else "NOT REVIEWED"
            )
            print(f"  {module}.{name}: {difference.source} -> {difference.destination} {status}")
        for problem in found.problems():
            print(f"  problem: {problem}")
        failed += not found.holds
    return 1 if failed else 0


def repository() -> Path:
    """The repository root this module lives in."""
    return Path(__file__).resolve().parents[1]


def resolve(repo: Path, revision: str) -> str:
    """The full commit hash of ``revision``."""
    shown = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode:
        raise AdoptionError(f"{revision!r} is not a commit of {repo}")
    return shown.stdout.strip()


def audit_revision(repo: Path, revision: str) -> RevisionAudit:
    """``revision``'s decision path against the working tree of ``repo``."""
    full = resolve(repo, revision)
    source, destination = git_sources(repo, full), tree_sources(repo)
    compared = audit(
        source,
        destination,
        DESTINATION_ENTRIES,
        source_entries=SOURCE_ENTRIES,
        leaves=AUDIT_LEAVES,
    )
    then = statements(Tree(source), *SOURCE_ARITHMETIC)
    now = statements(Tree(destination), *DESTINATION_ARITHMETIC)
    return RevisionAudit(
        full,
        compared,
        compared.unreviewed(REVIEWED),
        tuple(k for k in ARITHMETIC_KEYS if not then.get(k) or then.get(k) != now.get(k)),
        source(LOCK_FILE) == destination(LOCK_FILE),
    )


@functools.cache
def destination_fingerprint() -> str:
    """The fingerprint of the current tree's decision path."""
    reached = closure(
        Tree(tree_sources(repository())), FINGERPRINT_ENTRIES, leaves=FINGERPRINT_LEAVES
    )
    return fingerprint(reached)


# --- Adoption ------------------------------------------------------------------------------

#: The 8 fields of a saved ``power`` section.
_SAVED_FIELDS = frozenset(
    {
        "route",
        "planned",
        "basis",
        "asymptotic_share",
        "asymptotic",
        "asymptotic_part",
        "finite_part",
        "omitted",
    }
)
#: How much larger than a saved window mass the current bound may be: the saved figure is the
#: window's tails as summed, the current one that sum with its numerical error added.
_OMITTED_GROWTH = 1e-6


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest_of(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """The digest an adopted record cites its manifest by."""
    return _digest_of(manifest)


def _adopt_job(args: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """One saved record, verified against the current runtime: the enumeration it becomes and
    the plan section it is replaced by. Raises ``AdoptionError`` with the first disagreement."""
    from calibration import conversion_route as cr
    from increment.estimation.results import BINOMIAL_METHOD
    from increment.power._binomial import _UNIT_ROUNDOFF, _compounded, _inflation, refused

    cell, power = args
    lattice = cr._design_lattice(cell)
    where = f"design {cr._design_key(cell)}"
    if refused(lattice.key):
        raise AdoptionError(f"{where}: the runtime refuses this decision, which no replay covers")
    share, asymptotic, asymptotic_part, deferred = cr.delta_sums(cell)
    for name, now in (
        ("asymptotic_share", share),
        ("asymptotic", asymptotic),
        ("asymptotic_part", asymptotic_part),
    ):
        if power[name] != now:
            raise AdoptionError(f"{where}: {name} was saved as {power[name]!r} and sums to {now!r}")
    window_c, window_t = lattice.window_c, lattice.window_t
    omitted = cr._next_up(window_c.omitted + window_t.omitted)
    if not power["omitted"] <= omitted <= power["omitted"] * (1.0 + _OMITTED_GROWTH):
        raise AdoptionError(
            f"{where}: the window mass was saved as {power['omitted']!r}, and the windows of this "
            f"design bound {omitted!r}"
        )
    blocks = len(list(cr._row_blocks(window_c.size, lattice.x_t.size)))
    inflation = _inflation(
        window_c.error,
        window_t.error,
        _UNIT_ROUNDOFF,
        _compounded(window_c.size * window_t.size + blocks + 2),
    )
    return {
        "enumeration": asdict(
            cr.Enumeration(
                BINOMIAL_METHOD,
                lattice.floor,
                power["asymptotic_share"],
                power["asymptotic"],
                power["asymptotic_part"],
                power["finite_part"],
                omitted,
                inflation,
                deferred,
            )
        ),
        "planner": asdict(
            cr.Plan(
                "borderline_minimum",
                power["route"],
                power["basis"],
                power["planned"],
                0.0,
                1.0,
                0.0,
                closed_form=False,
                certified=False,
            )
        ),
    }


def _saved_records(
    path: Path, grid: str | None
) -> tuple[list[tuple[Any, dict[str, Any], str]], str]:
    """``(cell, saved power section, sha256 of its line)`` of every record of a saved ``bound``
    checkpoint, and its layout. Only the layout whose sections hold exactly the saved fields is
    read; a design keyed by an index is read through ``grid``."""
    from calibration import conversion_route as cr

    original = cr.bound_cells("original") if grid == "original" else ()
    records: list[tuple[Any, dict[str, Any], str]] = []
    layouts: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            record = json.loads(line)
            design, power = record["design"], record["power"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise AdoptionError(f"{where}: not a saved bound record") from error
        if record.keys() != {"design", "power"} or not (
            isinstance(power, dict) and power.keys() == _SAVED_FIELDS
        ):
            raise AdoptionError(f"{where}: its power section is not the saved layout's 8 fields")
        if type(design) is int:
            if not original or not 0 <= design < len(original):
                raise AdoptionError(f"{where}: index {design} needs --grid original")
            cell = original[design]
            layouts.add("indexed")
        else:
            try:
                cell = cr.MirrorCell("bound", *design)
            except TypeError as error:
                raise AdoptionError(f"{where}: design {design!r} is not a design key") from error
            layouts.add("keyed")
        records.append((cell, power, hashlib.sha256(line.encode()).hexdigest()))
    if len(layouts) != 1:
        raise AdoptionError(f"{path}: records of {sorted(layouts) or 'no'} layouts, expected one")
    return records, layouts.pop()


def adopt(
    source: Path,
    out: Path,
    *,
    revisions: Sequence[str],
    construction: str,
    law: str,
    grid: str | None = None,
    evidence: str = "",
    workers: int = 1,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Adopt the saved enumerations of ``source`` into ``out`` under a manifest.

    The caller names the runtime construction and routing law the records are to stand for;
    both must be the current ones. Every revision named must have the decision path of the
    current tree (``audit_revision``). Every record must then reproduce under the current
    runtime: the runtime must decide its design, and the routing share, delta-method parts and
    window mass must equal what was saved. The finite-sample part, the expensive one, is kept as
    saved; the planner section is replaced by one under the retired model, which a resumed run
    plans again. All of it or none: any disagreement leaves ``out`` unwritten.
    """
    from calibration import conversion_route as cr
    from increment.estimation.results import BINOMIAL_METHOD

    repo = repo or repository()
    if construction != BINOMIAL_METHOD:
        raise AdoptionError(
            f"the runtime construction is {BINOMIAL_METHOD!r}, not {construction!r}"
        )
    if law != route_law():
        raise AdoptionError(f"the routing law is {route_law()!r}, not {law!r}")
    if out.exists():
        raise AdoptionError(f"{out} exists; adoption writes a new file")
    if not revisions:
        raise AdoptionError("name the revision the campaign ran at (--revision)")
    audits = [audit_revision(repo, revision) for revision in revisions]
    problems = [problem for revision in audits for problem in revision.problems()]
    if problems:
        raise AdoptionError("the decision path is not the current one:\n" + "\n".join(problems))
    records, layout = _saved_records(source, grid)
    jobs = [(cell, power) for cell, power, _ in records]
    verified = cr._pool(workers).imap(_adopt_job, jobs) if workers > 1 else map(_adopt_job, jobs)
    adopted = list(verified)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "kind": "adoption",
        "source": {
            "file": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "layout": layout,
            "grid": grid,
            "records": len(records),
        },
        "campaign": {"revisions": [revision.revision for revision in audits], "evidence": evidence},
        "destination": {
            "construction": construction,
            "route_law": law,
            "head": head,
            "closure": destination_fingerprint(),
        },
        "audit": [
            {
                "revision": revision.revision,
                "source_closure": fingerprint(revision.audit.source),
                "destination_closure": fingerprint(revision.audit.destination),
                "reviewed": [
                    {
                        "definition": f"{d.definition[0]}.{d.definition[1]}",
                        "source": d.source,
                        "destination": d.destination,
                        "why": REVIEWED[d.definition].why,
                    }
                    for d in revision.audit.differences
                ],
                "arithmetic": list(ARITHMETIC_KEYS),
                "lock": LOCK_FILE,
            }
            for revision in audits
        ],
        "verified": [
            "the runtime decides the design",
            "asymptotic_share, asymptotic and asymptotic_part sum to the saved figures",
            "the saved window mass is within the current bound",
        ],
        "derived": ["omitted", "inflation", "deferred", "floor", "construction"],
        "planner": "saved plans carried under the retired model borderline_minimum",
        "records": {json.dumps(cr._design_key(cell)): sha for cell, _, sha in records},
        "enumerations": {
            json.dumps(cr._design_key(cell)): _digest_of(adoption["enumeration"])
            for (cell, _, _), adoption in zip(records, adopted, strict=True)
        },
    }
    digest_ = manifest_digest(manifest)
    lines = [json.dumps({"adoption": manifest})]
    lines += [
        json.dumps(
            {
                "design": cr._design_key(cell),
                **adoption,
                "adopted": digest_,
            }
        )
        for (cell, _, _), adoption in zip(records, adopted, strict=True)
    ]
    temporary = out.with_name(out.name + ".part")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, out)
    return manifest


def validate_adoption(
    manifests: Mapping[str, Mapping[str, Any]],
    cited: str,
    design: str,
    enumeration: Mapping[str, Any],
) -> None:
    """Refuse an adopted record whose manifest is absent, altered, for another runtime or
    routing law, silent about the design, or written against a decision path that is no longer
    the current one."""
    from increment.estimation.results import BINOMIAL_METHOD

    manifest = manifests.get(cited)
    if manifest is None:
        raise AdoptionError("the record cites an adoption manifest this file does not hold")
    destination = manifest["destination"]
    if destination["construction"] != BINOMIAL_METHOD or enumeration["construction"] != (
        BINOMIAL_METHOD
    ):
        raise AdoptionError("it was adopted for another runtime construction")
    if destination["route_law"] != route_law():
        raise AdoptionError("it was adopted for another routing law")
    if design not in manifest["records"]:
        raise AdoptionError("its manifest does not list this design")
    if manifest["enumerations"].get(design) != _digest_of(enumeration):
        raise AdoptionError("its enumeration is not the one its manifest adopted")
    if destination["closure"] != destination_fingerprint():
        raise AdoptionError(
            "the decision path has changed since it was adopted; audit and adopt again"
        )
