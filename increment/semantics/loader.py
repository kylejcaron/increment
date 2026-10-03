"""YAML loader for the semantic layer.

Accepts a single YAML file or a directory of ``*.yaml`` files (merged
by top-level list keys), validated via ``Definitions``. Every non-empty
``sql:`` value is admitted as exactly one read-only query via sqlglot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
import yaml.constructor
import yaml.resolver
from pydantic import ValidationError

from increment.errors import (
    CodedError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    _safe_error_value,
    refuse,
    warn,
)
from increment.errors import (
    DefinitionError as _DefinitionError,
)
from increment.semantics.models import (
    DEFINITION_REFUSALS,
    ConversionMetric,
    Definitions,
    MeanMetric,
)


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """`SafeLoader` that refuses duplicate mapping keys.

    PyYAML's default constructor keeps the LAST of any duplicate key
    with no warning, so ``dialect: duckdb`` followed by
    ``dialect: postgres`` silently loads as postgres, and a source with
    two ``sql:`` keys silently keeps only the second.
    """


def _construct_mapping_no_duplicates(
    # `deep` is positional-with-default in PyYAML's constructor protocol, which
    # calls this as construct_mapping(node, deep); the boolean-trap rule targets
    # our own API surface, not a third-party callback signature.
    loader: yaml.SafeLoader,
    node: yaml.Node,
    deep: bool = False,  # noqa: FBT001, FBT002
) -> dict[Any, Any]:
    seen: set[Any] = set()
    merge_keys = 0
    for key_node, _value_node in node.value:
        # SafeConstructor flattens `<<` below, so do not construct it as a key:
        # that fails and would make anchor reuse look duplicated. Count it anyway,
        # since repeated `<<` silently keeps only the last; one `<<` with a
        # sequence of anchors merges several.
        if key_node.tag == "tag:yaml.org,2002:merge":
            merge_keys += 1
            if merge_keys > 1:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate merge key '<<' -- merge several anchors with "
                    "one '<<: [*a, *b]' so their precedence is declared",
                    key_node.start_mark,
                )
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as exc:
            # An unhashable key (e.g. `? [a, b]`) is a malformed declaration;
            # raise the YAML error the loader path already turns into a
            # path-aware DefinitionError rather than a bare TypeError.
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep=deep)


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_duplicates
)

# The loader owns these definitions failures; the neutral error module only
# provides the primitive and rendering dispatch.
_MISSING_PATH = RefusalSpec(
    "definition.path.missing", _DefinitionError, template="path does not exist: {path}"
)
_MIGRATED_METRICS = RefusalSpec(
    "definition.migration.metrics",
    _DefinitionError,
    template="experiment '{experiment}': 'metrics' moved to plan.secondaries -- declare it as `plan: {{secondaries: [...]}}` instead",
    keys=frozenset({"path"}),
)
_MIGRATED_GUARDRAILS = RefusalSpec(
    "definition.migration.guardrails",
    _DefinitionError,
    template="experiment '{experiment}': 'guardrails' moved to plan.guardrails -- declare it as `plan: {{guardrails: [...]}}` instead",
    keys=frozenset({"path"}),
)
_DUPLICATES = DEFINITION_REFUSALS["definition.duplicates"]


def _render_validation(*, path: str, errors: Sequence[Mapping[str, object]], route: str) -> str:
    lines = "".join(
        f"\n  {error['field']}: {error['reason']} [type={error['type']}, input={error['input']!r}]"
        for error in errors
    )
    return f"{len(errors)} validation error(s) in definitions {path}:{lines}\n{route}"


_VALIDATION = RefusalSpec("definition.validation", _DefinitionError, _render_validation)
_DUPLICATE_SCALAR = RefusalSpec(
    "definition.duplicate_scalar",
    _DefinitionError,
    template="scalar key {field!r} is declared in more than one YAML file {files!r}; {route}",
    keys=frozenset({"path"}),
)
_YAML = RefusalSpec(
    "definition.yaml",
    _DefinitionError,
    template="YAML parse error in {path}: {error}; {route}",
    keys=frozenset({"line", "column"}),
)
_ROOT = RefusalSpec(
    "definition.root",
    _DefinitionError,
    template="YAML root in {path} must be a {expected}, got {value_type}; {route}",
)
_SECTION = RefusalSpec(
    "definition.section",
    _DefinitionError,
    template="definition field {field} in {path} must be a {expected}, got {value_type}; {route}",
)
_SQL = RefusalSpec(
    "definition.sql",
    _DefinitionError,
    template="{label}: SQL admission failed with {error_type}: {error}; {route}",
    keys=frozenset({"path"}),
)
_VARIABLE_WINDOW_DAYS_NONE_WARNING = WarningSpec(
    "definition.metric.variable_window_days_none",
    IncrementWarning,
    lambda *, name: (
        f"metric '{name}' has window_days=None (variable per-unit window); "
        f"specify window_days if a fixed window is intended"
    ),
)
_NO_YAML_FILES_WARNING = WarningSpec(
    "definition.directory.no_yaml_files",
    IncrementWarning,
    lambda *, path: (
        f"definitions directory {path} contains no .yaml/.yml files -- "
        f"loading an empty semantic layer; check the path if this is "
        f"unexpected"
    ),
)

# Names of top-level list-valued keys that the loader merges across files
_MERGE_LISTS = {"fact_sources", "dim_sources", "exposures", "metrics", "experiments"}


def load(path: str | Path) -> Definitions:
    """Load and validate a definitions YAML file or directory.

    Accepts a single file or a directory of ``**/*.yaml``/``**/*.yml``
    files, merged and validated. Raises ``DefinitionError`` on invalid
    YAML, invalid SQL, duplicate names, or dangling references.
    """
    path_obj = Path(path)

    if path_obj.is_file():
        raw, origins = _load_single(path_obj)
    elif path_obj.is_dir():
        raw, origins = _load_dir(path_obj)
    else:
        refuse(_MISSING_PATH, path=str(path_obj))

    dialect = raw.get("dialect")

    # Validate SQL before model validation: fact sources, dim sources, and
    # exposures are all user-supplied SQL and go through the same check.
    for fs in raw.get("fact_sources", []):
        sql = fs.get("sql", "")
        _check_sql(dialect, sql, f"fact source '{fs.get('name', '?')}'", path_obj)
    for ds in raw.get("dim_sources", []):
        _check_sql(dialect, ds.get("sql", ""), f"dim source '{ds.get('name', '?')}'", path_obj)
    for ex in raw.get("exposures", []):
        sql = ex.get("sql") or ""
        _check_sql(dialect, sql, f"exposure '{ex.get('name', '?')}'", path_obj)

    # Reject `metrics:`/`guardrails:` directly on an experiment (they belong under
    # `plan:`). Checked on the raw dict -- once validated, `extra="forbid"` alone
    # would give a far less helpful message.
    for exp in raw.get("experiments", []):
        if "metrics" in exp:
            refuse(
                _MIGRATED_METRICS,
                experiment=exp.get("name", "?"),
                path=str(path_obj),
            )
        if "guardrails" in exp:
            refuse(
                _MIGRATED_GUARDRAILS,
                experiment=exp.get("name", "?"),
                path=str(path_obj),
            )

    # Duplicate-name pre-check (so the error includes file names)
    dup_errors = _check_duplicates(raw, origins)
    if dup_errors:
        refuse(
            _DUPLICATES,
            message="; ".join(dup_errors),
            path=str(path_obj),
            sources=origins,
        )

    try:
        defs = Definitions.model_validate(raw)
    except _DefinitionError as exc:
        raise _DefinitionError(
            exc.message, code=exc.code, context={**exc.context, "path": str(path_obj)}
        ) from exc
    except InvalidRequestError as exc:
        raise _DefinitionError(
            str(exc), code=exc.code, context={**dict(exc.context), "path": str(path_obj)}
        ) from exc
    except ValidationError as exc:
        errors = tuple(
            {
                "field": ".".join(str(part) for part in error["loc"]) or None,
                "type": error["type"],
                "reason": error["msg"],
                "input": _safe_error_value(error.get("input")),
            }
            for error in exc.errors(include_url=False)
        )
        try:
            refuse(
                _VALIDATION,
                path=str(path_obj),
                errors=errors,
                route="repair the listed definition fields and reload",
            )
        except _DefinitionError as error:
            raise error from exc

    # day_boundary inheritance lives on Definitions itself
    # (`_inherit_day_boundary`), so model_validate and load() agree.

    # Warn when variable-window metrics omit window_days
    for m in defs.metrics:
        if isinstance(m, MeanMetric | ConversionMetric) and m.window_days is None:
            warn(_VARIABLE_WINDOW_DAYS_NONE_WARNING, context={"name": m.name}, stacklevel=2)
    return defs


# Internal helpers
# ---------------------------------------------------------------------------


def _load_single(path: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    data = _read_yaml(path)
    _validate_merge_sections(data, path)
    origins: dict[str, list[str]] = {}
    _record_origins(data, str(path), origins)
    return data, origins


def _load_dir(path: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    merged: dict[str, Any] = {}
    origins: dict[str, list[str]] = {}
    scalar_files: dict[str, str] = {}

    files = sorted({*path.rglob("*.yaml"), *path.rglob("*.yml")})
    if not files:
        # A warning, not a refusal: an empty directory is a valid workflow,
        # but is more often a typo'd path whose symptom lands far from the cause.
        warn(_NO_YAML_FILES_WARNING, context={"path": path}, stacklevel=3)
    for yaml_file in files:
        data = _read_yaml(yaml_file)
        _validate_merge_sections(data, yaml_file)
        for key, value in data.items():
            if key in _MERGE_LISTS:
                existing = merged.setdefault(key, [])
                if isinstance(value, list):
                    existing.extend(value)
            else:
                if key in merged:
                    refuse(
                        _DUPLICATE_SCALAR,
                        path=str(yaml_file),
                        field=key,
                        files=(scalar_files[key], str(yaml_file)),
                        route="declare the scalar key in exactly one YAML file",
                    )
                merged[key] = value
                scalar_files[key] = str(yaml_file)
        _record_origins(data, str(yaml_file), origins)

    return merged, origins


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        try:
            data = yaml.load(f, Loader=_DuplicateKeySafeLoader)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            try:
                refuse(
                    _YAML,
                    path=str(path),
                    line=None if mark is None else mark.line + 1,
                    column=None if mark is None else mark.column + 1,
                    error=str(exc),
                    route="fix the YAML at the reported position and reload",
                )
            except _DefinitionError as error:
                raise error from exc
    if not isinstance(data, dict):
        refuse(
            _ROOT,
            path=str(path),
            expected="mapping",
            value_type=type(data).__name__,
            route="make the YAML document a mapping of definition sections",
        )
    return data


def _validate_merge_sections(data: dict[str, Any], path: Path) -> None:
    for key in _MERGE_LISTS:
        if key not in data:
            continue
        if not isinstance(data[key], list):
            refuse(
                _SECTION,
                path=str(path),
                field=key,
                expected="list",
                value_type=type(data[key]).__name__,
                route=f"declare {key} as a YAML list of mappings",
            )
        for index, item in enumerate(data[key]):
            if not isinstance(item, dict):
                refuse(
                    _SECTION,
                    path=str(path),
                    field=f"{key}[{index}]",
                    expected="mapping",
                    value_type=type(item).__name__,
                    route=f"declare each {key} item as a mapping",
                )


def _record_origins(data: dict, file_path: str, origins: dict[str, list[str]]):
    """Track which file each named element came from."""
    _track_names(data.get("fact_sources", []), "fact_source", file_path, origins)
    _track_names(data.get("dim_sources", []), "dim_source", file_path, origins)
    _track_names(data.get("exposures", []), "exposure", file_path, origins)
    _track_names(data.get("metrics", []), "metric", file_path, origins)
    _track_names(data.get("experiments", []), "experiment", file_path, origins)


def _track_names(
    items: object,
    kind: str,
    file_path: str,
    origins: dict[str, list[str]],
):
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            key = f"{kind}:{name}"
            origins.setdefault(key, []).append(file_path)


def _check_duplicates(
    raw: dict[str, Any],
    origins: dict[str, list[str]],
) -> list[str]:
    errors: list[str] = []
    _check_category_duplicates(raw.get("fact_sources", []), "fact_source", origins, errors)
    _check_category_duplicates(raw.get("dim_sources", []), "dim_source", origins, errors)
    _check_category_duplicates(raw.get("exposures", []), "exposure", origins, errors)
    _check_category_duplicates(raw.get("metrics", []), "metric", origins, errors)
    _check_category_duplicates(raw.get("experiments", []), "experiment", origins, errors)
    _check_fact_name_duplicates(raw.get("fact_sources", []), origins, errors)

    # Property-name duplicates within a fact source (names file origins)
    _check_property_name_duplicates(raw.get("fact_sources", []), origins, errors)
    return errors


def _check_category_duplicates(
    items: object,
    kind: str,
    origins: dict[str, list[str]],
    errors: list[str],
):
    if not isinstance(items, list):
        return
    seen: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        key = f"{kind}:{name}"
        if name in seen:
            files = _files_for(origins, key)
            errors.append(
                f"duplicate {kind} name '{name}' appears in files: "
                + (", ".join(sorted(files)) if files else "(same file)")
            )
        seen[name] = origins.get(key, [])


def _check_fact_name_duplicates(
    sources: object,
    origins: dict[str, list[str]],
    errors: list[str],
):
    """Check that no fact name appears in two different fact sources."""
    if not isinstance(sources, list):
        return
    fact_to_sources: dict[str, list[str]] = defaultdict(list)
    for fs in sources:
        if not isinstance(fs, Mapping):
            continue
        fs_name = fs.get("name", "?")
        if not isinstance(fs_name, str) or not fs_name:
            fs_name = "?"
        facts = fs.get("facts", [])
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, Mapping):
                continue
            fname = fact.get("name")
            if isinstance(fname, str) and fname:
                fact_to_sources[fname].append(fs_name)

    for fname, srcs in fact_to_sources.items():
        if len(srcs) > 1:
            # Find files that contain these sources
            files: set[str] = set()
            for src_name in srcs:
                key = f"fact_source:{src_name}"
                files.update(origins.get(key, []))
            errors.append(
                f"duplicate fact name '{fname}' across sources "
                f"{srcs}; appears in files: " + (", ".join(sorted(files)) if files else "(unknown)")
            )


def _check_property_name_duplicates(
    sources: object,
    origins: dict[str, list[str]],
    errors: list[str],
):
    """Check for duplicate property names within each fact source, naming file origins."""
    if not isinstance(sources, list):
        return
    for fs in sources:
        if not isinstance(fs, Mapping):
            continue
        fs_name = fs.get("name", "?")
        if not isinstance(fs_name, str) or not fs_name:
            fs_name = "?"
        properties = fs.get("properties", [])
        if not isinstance(properties, list):
            continue
        seen: set[str] = set()
        for prop in properties:
            if not isinstance(prop, Mapping):
                continue
            pname = prop.get("name")
            if not isinstance(pname, str) or not pname:
                continue
            if pname in seen:
                key = f"fact_source:{fs_name}"
                files = _files_for(origins, key)
                errors.append(
                    f"duplicate property name '{pname}' in fact source '{fs_name}' "
                    "appears in files: " + (", ".join(sorted(files)) if files else "(same file)")
                )
            seen.add(pname)


def _files_for(origins: dict[str, list[str]], key: str) -> list[str]:
    return origins.get(key, [])


#: Function names that acquire locks or perform file/session side effects
#: across the dialects this project targets. A plain forbidden-statement
#: check (INSERT/UPDATE/... at the statement level) never sees these --
#: they parse as ordinary function calls inside an otherwise-safe SELECT.
_SIDE_EFFECTING_FUNCTIONS = frozenset(
    {
        # Advisory/session locks
        "pg_advisory_lock",
        "pg_advisory_lock_shared",
        "pg_advisory_unlock",
        "pg_advisory_unlock_all",
        "pg_advisory_unlock_shared",
        "pg_advisory_xact_lock",
        "pg_advisory_xact_lock_shared",
        "pg_try_advisory_lock",
        "pg_try_advisory_lock_shared",
        "pg_try_advisory_xact_lock",
        "pg_try_advisory_xact_lock_shared",
        "get_lock",
        "release_lock",
        "release_all_locks",
        "is_free_lock",
        "is_used_lock",
        # File / process writes
        "lo_export",
        "lo_import",
        "lo_create",
        "lo_creat",
        "lo_unlink",
        "lowrite",
        "lo_put",
        "lo_from_bytea",
        "lo_truncate",
        "lo_truncate64",
        "load_file",
        "write_csv",
        "write_parquet",
        "write_json",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "system",
        "xp_cmdshell",
        # Server / session state
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_promote",
        "pg_rotate_logfile",
        "pg_notify",
        "dblink",
        "dblink_exec",
        "dblink_send_query",
        "dblink_connect",
        "dblink_disconnect",
        "set_config",
        "setval",
        "nextval",
        "sleep",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "benchmark",
        # Snowflake session/admin operations
        "system$cancel_query",
        "system$abort_session",
        "system$wait",
        # MySQL replication waits
        "master_pos_wait",
        "wait_for_executed_gtid_set",
    }
)

_SQL_DIALECT = RefusalSpec(
    "definition.sql.dialect",
    InvalidRequestError,
    template=(
        "{label}: unknown SQL dialect {dialect!r} -- admission cannot be "
        "verified against a parser that does not exist; {route}"
    ),
)
_SQL_PARSE = RefusalSpec(
    "definition.sql.parse",
    InvalidRequestError,
    template="invalid SQL in {label}: {error}; {route}",
    keys=frozenset({"dialect", "line", "column"}),
)
_SQL_STATEMENT_COUNT = RefusalSpec(
    "definition.sql.statement_count",
    InvalidRequestError,
    template=(
        "{label} must contain exactly one read-only query; parsed {statements} "
        "statement(s), {empty} of them empty; {route}"
    ),
    keys=frozenset({"dialect"}),
)
_SQL_NOT_READ_ONLY = RefusalSpec(
    "definition.sql.not_read_only",
    InvalidRequestError,
    template=(
        "{label} must contain exactly one read-only query; found {violation} {construct!r}; {route}"
    ),
    keys=frozenset({"dialect"}),
)
_DIALECT_GUESS_PARSE_FAILED_WARNING = WarningSpec(
    "definition.sql.dialect_guess_parse_failed",
    IncrementWarning,
    lambda *, label, parse_exc: (
        f"could not parse SQL in {label} with a guessed "
        f"dialect (no `dialect` declared in definitions): {parse_exc}"
    ),
)


def admit_read_only_sql(sql: str, *, dialect: str | None, label: str) -> None:
    """Admit exactly one side-effect-free SQL query.

    ``sqlglot`` is intentionally imported here, rather than at module load
    time, so importing the semantic layer does not require the optional SQL
    parser until SQL validation is requested.
    """
    import sqlglot
    from sqlglot import exp

    if dialect is not None and dialect not in sqlglot.Dialect.classes:
        refuse(
            _SQL_DIALECT,
            label=label,
            dialect=dialect,
            route="declare a dialect sqlglot recognizes, or omit it to use the generic parser",
        )

    try:
        statements = sqlglot.parse(sql, dialect=dialect)
    except sqlglot.errors.SqlglotError as exc:
        # refuse() does not chain __cause__; sqlglot's own message carries the parse detail.
        errors = getattr(exc, "errors", None) or ({},)
        refuse(
            _SQL_PARSE,
            label=label,
            dialect=dialect,
            line=errors[0].get("line"),
            column=errors[0].get("col"),
            error=str(exc),
            route="repair the SQL syntax for the declared or inferred dialect",
        )

    if len(statements) != 1 or statements[0] is None:
        refuse(
            _SQL_STATEMENT_COUNT,
            label=label,
            dialect=dialect,
            statements=len(statements),
            empty=sum(statement is None for statement in statements),
            route="submit a single SELECT or WITH query with no further statements",
        )

    statement = statements[0]
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
        exp.Set,
        exp.Into,
        exp.Lock,
    )
    locking_hints = {
        "HOLDLOCK",
        "PAGLOCK",
        "READCOMMITTEDLOCK",
        "ROWLOCK",
        "TABLOCK",
        "REPEATABLEREAD",
        "SERIALIZABLE",
        "TABLOCKX",
        "UPDLOCK",
        "XLOCK",
    }
    forbidden_clause = next(
        (type(node).__name__ for node in map(statement.find, forbidden) if node is not None),
        None,
    )
    locking_hint = next(
        (
            hint.name.upper()
            for hint_group in statement.find_all(exp.WithTableHint)
            for hint in hint_group.expressions
            if hint.name.upper() in locking_hints
        ),
        None,
    )

    def function_name(fn: exp.Func) -> str:
        if isinstance(fn, exp.Anonymous):
            return fn.name.lower()
        sql_name = getattr(fn, "sql_name", None)
        if callable(sql_name):
            return str(sql_name()).lower()
        return type(fn).__name__.lower()

    side_effecting_function = next(
        (
            name
            for fn in statement.find_all(exp.Func)
            if (name := function_name(fn)) in _SIDE_EFFECTING_FUNCTIONS
        ),
        None,
    )
    if not isinstance(statement, exp.Query):
        violation, construct = "non-query statement", type(statement).__name__
    elif forbidden_clause is not None:
        violation, construct = "write or lock clause", forbidden_clause
    elif locking_hint is not None:
        violation, construct = "locking table hint", locking_hint
    elif side_effecting_function is not None:
        violation, construct = "side-effecting function", side_effecting_function
    else:
        return
    refuse(
        _SQL_NOT_READ_ONLY,
        label=label,
        dialect=dialect,
        violation=violation,
        construct=construct,
        route="rewrite the source as one side-effect-free SELECT or WITH query",
    )


def _check_sql(dialect: str | None, sql: Any, label: str, path: Path):
    """Validate definition SQL, retaining guessed-dialect parse warnings."""
    if sql is not None and not isinstance(sql, str):
        refuse(
            _SECTION,
            path=str(path),
            field=f"{label}.sql",
            expected="string",
            value_type=type(sql).__name__,
            route="set sql to a query string",
        )
    if not sql or not sql.strip():
        return
    try:
        admit_read_only_sql(sql, dialect=dialect, label=label)
    except ValueError as exc:
        if dialect is None:
            import sqlglot

            try:
                sqlglot.parse(sql, dialect=dialect)
            except sqlglot.errors.SqlglotError as parse_exc:
                warn(
                    _DIALECT_GUESS_PARSE_FAILED_WARNING,
                    context={"label": label, "parse_exc": str(parse_exc)},
                    stacklevel=2,
                )
        if isinstance(exc, CodedError):
            raise _DefinitionError(
                str(exc), code=exc.code, context={**exc.context, "path": str(path)}
            ) from exc
        try:
            refuse(
                _SQL,
                path=str(path),
                label=label,
                error_type=type(exc).__name__,
                error=str(exc),
                route="repair the SQL definition and reload",
            )
        except _DefinitionError as error:
            raise error from exc


# Importing an ibis SQL backend registers its sqlglot dialect, so `con.name`
# resolves directly for nearly every backend. Map the exceptions: ibis registers
# `singlestoredb` as `singlestore`; sqlglot names `mssql`/`pyspark` `tsql`/`spark`.
# Unregistered names make `resolve_execution_dialect` return None, not refuse.
_IBIS_TO_SQLGLOT_DIALECT = {
    "mssql": "tsql",
    "pyspark": "spark",
    "singlestoredb": "singlestore",
}


def resolve_execution_dialect(defs: Definitions, con: Any) -> str | None:
    """The sqlglot dialect name matching *con*'s own SQL dialect, or `None`
    when sqlglot has no dialect for it at all. Returns `defs.dialect`
    unchanged when declared (already validated at `load()` time; this is a
    no-op read, not a re-resolution). When undeclared, resolves to the
    connection's own backend name -- mapped through
    `_IBIS_TO_SQLGLOT_DIALECT` for the three confirmed ibis/sqlglot name
    mismatches -- or `None` if that name is not a sqlglot dialect at all:
    never a refusal, so a backend sqlglot has never heard of keeps admitting
    SQL exactly as it does today (the generic-dialect fallback
    `verify_sql_admission_matches_execution` and `load()` both already use
    for `dialect=None`).
    """
    if defs.dialect is not None:
        return defs.dialect
    import sqlglot

    candidate = _IBIS_TO_SQLGLOT_DIALECT.get(con.name, con.name)
    return candidate if candidate in sqlglot.Dialect.classes else None


def verify_sql_admission_matches_execution(defs: Definitions, con: Any) -> None:
    """Re-validate every fact/dim/exposure SQL source's admission against
    the connection's own dialect when `dialect:` was left undeclared, so
    admission and execution agree by construction wherever that dialect is
    known. A no-op when `dialect:` is declared -- `load()` already
    validated every source against it.
    """
    if defs.dialect is not None:
        return
    dialect = resolve_execution_dialect(defs, con)  # None is a valid, expected fallback
    for fact_source in defs.fact_sources:
        if fact_source.sql.strip():
            admit_read_only_sql(
                fact_source.sql, dialect=dialect, label=f"fact source '{fact_source.name}'"
            )
    for dim_source in defs.dim_sources:
        if dim_source.sql.strip():
            admit_read_only_sql(
                dim_source.sql, dialect=dialect, label=f"dim source '{dim_source.name}'"
            )
    for exposure in defs.exposures:
        if exposure.sql is not None and exposure.sql.strip():
            admit_read_only_sql(exposure.sql, dialect=dialect, label=f"exposure '{exposure.name}'")
