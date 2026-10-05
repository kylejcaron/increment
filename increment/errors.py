"""Stable, coded error primitives shared across increment's public APIs."""

from __future__ import annotations

import inspect
import os
import string
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import KW_ONLY, dataclass
from functools import update_wrapper
from types import MappingProxyType, NoneType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    NoReturn,
    Self,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticSerializationError


def _freeze(value: object) -> object:
    """Recursively copy container values into immutable snapshots."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, (np.ndarray, np.generic)):
        # Arrays copy into plain tuples so the caller's buffer cannot alias
        # the snapshot; numpy scalars become Python scalars.
        return _freeze(value.tolist())
    return value


def _safe_error_value(value: object) -> object:
    """Keep refusal contexts bounded and pickle-safe for worker transport."""
    if type(value) is str:
        return value[:256]
    if type(value) is bytes:
        return value[:256]
    if type(value) is int:
        # About 240 decimal digits; larger ints are described, never stringified.
        return value if value.bit_length() <= 800 else f"<int of {value.bit_length()} bits>"
    if type(value) is float:
        return value
    if type(value) is bool:
        return value
    if value is None:
        return None
    try:
        return repr(value)[:256]
    except Exception:
        # A value whose own repr fails is described by its type alone.
        return f"<{type(value).__name__} with a failing repr>"


def _thaw(value: object) -> object:
    """Inverse of `_freeze`: unwind `MappingProxyType` into plain `dict`
    recursively, leaving already-picklable tuples/frozensets alone.

    Pickle's only actual blocker among `_freeze`'s outputs is
    `MappingProxyType` (tuples, frozensets, and scalars all pickle fine),
    so this is the minimal thaw a `CodedError.__getstate__` needs.
    """
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw(item) for item in value)
    return value


def _new_coded_error(cls: type[CodedError]) -> CodedError:
    """Allocate a bare instance for unpickling; `__setstate__` restores
    every attribute, so `__init__` (whose signature varies by subclass --
    see `DefinitionError`, `ArtifactContractError`) is never re-invoked."""
    return cls.__new__(cls)


class CodedError(ValueError):
    """A public error carrying a stable code and immutable call context."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, object],
    ) -> None:
        self._code = code
        self._context = cast("Mapping[str, object]", _freeze(context))
        super().__init__(message)

    @property
    def code(self) -> str:
        """Stable, machine-matchable refusal code; fixed at construction."""
        return self._code

    @property
    def context(self) -> Mapping[str, object]:
        """Immutable snapshot of the refusal's call context."""
        return self._context

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[type[CodedError]], CodedError], tuple[type[CodedError]], dict[str, object]
    ]:
        # BaseException.__reduce__ calls `cls(*self.args)`, which cannot supply the
        # keyword-only code/context or pickle a MappingProxyType context. Rebuild via
        # __new__ and __setstate__ so every subclass pickles regardless of __init__.
        return (_new_coded_error, (type(self),), self.__getstate__())

    def __getstate__(self) -> dict[str, object]:
        state = {key: _thaw(value) for key, value in self.__dict__.items()}
        state["_pickled_args"] = self.args
        return state

    def __setstate__(self, state: dict[str, object] | None) -> None:
        state = dict(state) if state else {}
        self.args = cast("tuple[Any, ...]", state.pop("_pickled_args", ()))
        for key, value in state.items():
            setattr(self, key, _freeze(value))


class CapabilityError(CodedError):
    """Raised when a source cannot honestly provide a requested capability."""


class InvalidRequestError(CodedError):
    """Raised when a public request is invalid."""


class WireFormatError(CodedError):
    """Raised when serialized source data has an invalid wire format."""


def _copy_sources(value: object) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    copied: dict[str, tuple[str, ...]] = {}
    for name, files in value.items():
        if (
            isinstance(name, str)
            and isinstance(files, (list, tuple))
            and all(isinstance(item, str) for item in files)
        ):
            copied[name] = cast("tuple[str, ...]", tuple(files))
    return MappingProxyType(copied)


class DefinitionError(CodedError):
    """A definition YAML parse, model validation, or reference failed.

    Carries ``message``, the file or directory ``path`` given to the
    loader, and a copied ``sources`` mapping for duplicate-name errors.
    """

    sources: Mapping[str, tuple[str, ...]]
    message: str

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, object],
    ) -> None:
        raw_path = context.get("path")
        self.path = raw_path if isinstance(raw_path, str) else None
        self.sources = _copy_sources(context.get("sources"))
        self.message = message
        copied_context = dict(context)
        if "sources" in copied_context:
            copied_context["sources"] = self.sources
        super().__init__(message, code=code, context=copied_context)


class UnsupportedRequestError(CodedError, NotImplementedError):
    """Raised when a recognized request is not supported."""


def _new_coded_warning(cls: type[IncrementWarning]) -> IncrementWarning:
    """Allocate a bare instance for unpickling; `__setstate__` restores
    every attribute, mirroring `_new_coded_error`."""
    return cls.__new__(cls)


class IncrementWarning(UserWarning):
    """A public warning carrying a stable code and immutable call context.

    Mirrors `CodedError`: `.code` is fixed at construction and `.context`
    is an immutable snapshot. A category that must also match
    `RuntimeWarning`/`DeprecationWarning` for existing `filterwarnings`
    calls should multiply-inherit from this class and that category
    (see `IncrementRuntimeWarning`, `IncrementDeprecationWarning`) rather
    than replacing it.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, object] | None = None,
    ) -> None:
        self._code = code
        self._context = cast("Mapping[str, object]", _freeze(context or {}))
        super().__init__(message)

    @property
    def code(self) -> str:
        """Stable, machine-matchable warning code; fixed at construction."""
        return self._code

    @property
    def context(self) -> Mapping[str, object]:
        """Immutable snapshot of the warning's call context."""
        return self._context

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[type[IncrementWarning]], IncrementWarning],
        tuple[type[IncrementWarning]],
        dict[str, object],
    ]:
        return (_new_coded_warning, (type(self),), self.__getstate__())

    def __getstate__(self) -> dict[str, object]:
        state = {key: _thaw(value) for key, value in self.__dict__.items()}
        state["_pickled_args"] = self.args
        return state

    def __setstate__(self, state: dict[str, object] | None) -> None:
        state = dict(state) if state else {}
        self.args = cast("tuple[Any, ...]", state.pop("_pickled_args", ()))
        for key, value in state.items():
            setattr(self, key, _freeze(value))


class IncrementRuntimeWarning(IncrementWarning, RuntimeWarning):
    """A coded warning that also satisfies `filterwarnings(category=RuntimeWarning)`."""


class IncrementDeprecationWarning(IncrementWarning, DeprecationWarning):
    """A coded warning that also satisfies `filterwarnings(category=DeprecationWarning)`."""


Renderer = Callable[..., str]


def _unset_render(**_: object) -> str:
    raise AssertionError("RefusalSpec.render was never set")


@dataclass(frozen=True, slots=True)
class _TemplateRenderer:
    """Picklable `str.format` renderer with a required-context-key check."""

    code: str
    template: str
    keys: frozenset[str]

    def __call__(self, /, **context: object) -> str:
        missing = self.keys - context.keys()
        if missing:
            raise AssertionError(
                f"refusal {self.code!r} missing required context keys: {sorted(missing)!r}"
            )
        return self.template.format(**context)


@dataclass(frozen=True, slots=True)
class RefusalSpec:
    """Stable code, exception type, and renderer for one refusal.

    Exactly one of `render` (a callable, for text that needs computation)
    or `template` (a `str.format` string, for plain interpolation) must be
    set. `keys` is the template's own `{placeholder}` names, unioned with
    any explicitly declared extras; a `render`-only spec's `keys` stays
    empty -- its own call signature is its drift guard.
    """

    code: str
    error_type: type[CodedError]
    render: Renderer = _unset_render
    _: KW_ONLY
    template: str | None = None
    keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        has_template = self.template is not None
        # A carried _TemplateRenderer (from dataclasses.replace or a pickle
        # round trip) is rebuilt below, not a user-supplied render.
        carried = isinstance(self.render, _TemplateRenderer)
        has_render = self.render is not _unset_render and not carried
        if has_template == has_render:
            raise AssertionError(
                f"RefusalSpec {self.code!r} needs exactly one of template or render, "
                f"got template={self.template!r} render={self.render if has_render else None!r}"
            )
        if has_template:
            placeholders = frozenset(
                field_name.split("[")[0].split(".")[0]
                for _, field_name, _, _ in string.Formatter().parse(self.template)
                if field_name
            )
            keys = placeholders | self.keys
            object.__setattr__(self, "keys", keys)
            object.__setattr__(self, "render", _TemplateRenderer(self.code, self.template, keys))
            return
        params = inspect.signature(self.render).parameters.values()
        required_positional_only = [
            p.name
            for p in params
            if p.kind is inspect.Parameter.POSITIONAL_ONLY and p.default is inspect.Parameter.empty
        ]
        if required_positional_only:
            raise AssertionError(
                f"RefusalSpec {self.code!r} cannot bind required positional-only "
                f"renderer parameters by keyword: {required_positional_only!r}"
            )


def refuse(spec: RefusalSpec, /, **context: object) -> NoReturn:
    """Render one refusal and raise its declared coded error type."""
    message = spec.render(**context)
    raise spec.error_type(message, code=spec.code, context=context)


def refusals(
    error_type: type[CodedError],
    entries: Mapping[str, str | tuple[str, tuple[str, ...]] | RefusalSpec],
) -> dict[str, RefusalSpec]:
    """Build a `code -> RefusalSpec` registry. Each entry is a plain
    `str.format` template, a `(template, extra_keys)` pair (for a template
    whose caller-supplied keys are not all used as placeholders), or a
    prebuilt `RefusalSpec` (passed through unchanged, its own `code` must
    match the registry key)."""
    registry: dict[str, RefusalSpec] = {}
    for code, entry in entries.items():
        if isinstance(entry, RefusalSpec):
            assert entry.code == code, f"registry key {code!r} != spec code {entry.code!r}"
            registry[code] = entry
        elif isinstance(entry, tuple):
            template, extra_keys = entry
            registry[code] = RefusalSpec(
                code, error_type, template=template, keys=frozenset(extra_keys)
            )
        else:
            registry[code] = RefusalSpec(code, error_type, template=entry)
    return registry


def raiser(registry: Mapping[str, RefusalSpec]) -> Callable[..., NoReturn]:
    """Build a module's `_raise(code, **context)` from its registry."""

    def _raise(code: str, /, **context: object) -> NoReturn:
        refuse(registry[code], **context)

    return _raise


@dataclass(frozen=True, slots=True)
class WarningSpec:
    """Stable code, warning type, and renderer for one coded warning."""

    code: str
    warning_type: type[IncrementWarning]
    render: Renderer

    def __post_init__(self) -> None:
        params = inspect.signature(self.render).parameters.values()
        required_positional_only = [
            p.name
            for p in params
            if p.kind is inspect.Parameter.POSITIONAL_ONLY and p.default is inspect.Parameter.empty
        ]
        if required_positional_only:
            raise AssertionError(
                f"WarningSpec {self.code!r} cannot bind required positional-only "
                f"renderer parameters by keyword: {required_positional_only!r}"
            )


#: ``skip_file_prefixes`` that attribute a warning to the first frame outside this package: the
#: caller of the public API, however many private layers the call passes through.
PACKAGE_FRAMES: tuple[str, ...] = (os.path.join(os.path.dirname(os.path.abspath(__file__)), ""),)


def warn(
    spec: WarningSpec,
    /,
    *,
    stacklevel: int = 2,
    skip_file_prefixes: tuple[str, ...] = (),
    context: Mapping[str, object] | None = None,
) -> None:
    """Render one coded warning and emit it via `warnings.warn`.

    `context` is a single explicit mapping (never `**kwargs`) so a local
    `_warn(code, /, **context)` forwarding helper can pass it straight
    through as `context=context` without a keyword-splat colliding with
    `stacklevel`/`skip_file_prefixes` -- the shape a type checker cannot
    otherwise rule out through a `**kwargs: object` splat.

    `stacklevel` means the same thing it would at a direct `warnings.warn`
    call: this function adds its own frame to `warnings.warn`'s count
    internally (`stacklevel + 1`), so a caller need not account for going
    through `warn()` itself. A local `_warn(code, /, *, stacklevel=2,
    **context)` helper that forwards to this function should do the same
    (`warn(spec, stacklevel=stacklevel + 1, context=context)`) to absorb
    its own frame -- callers at the original `warnings.warn` sites then
    keep their original numeric `stacklevel` literal unchanged.
    """
    ctx = context or {}
    message = spec.render(**ctx)
    instance = spec.warning_type(message, code=spec.code, context=ctx)
    if skip_file_prefixes:
        warnings.warn(instance, skip_file_prefixes=(*skip_file_prefixes, __file__))
    else:
        warnings.warn(instance, stacklevel=stacklevel + 1)


def unwrap_coded(exc: ValidationError) -> None:
    """Re-raise the first coded refusal buried in a pydantic ``ValidationError``.

    Use this at schema boundaries such as ``TypeAdapter`` or an ordinary
    ``BaseModel`` containing a ``CodedModel``. Returns (does nothing) if no
    coded error is found, so the caller can re-raise the original
    ``ValidationError`` unchanged.
    """
    for err in exc.errors():
        inner = err.get("ctx", {}).get("error")
        if isinstance(inner, CodedError):
            raise inner from exc


def unwrap_coded_serialization(exc: PydanticSerializationError) -> None:
    """Re-raise the coded refusal a field serializer raised inside a model dump.

    pydantic wraps every serializer exception; a python-mode dump chains the
    original as ``__cause__``. Returns (does nothing) when the cause is not a
    coded error, so the caller can re-raise the original unchanged.
    """
    if isinstance(exc.__cause__, CodedError):
        raise exc.__cause__ from exc


# Options model_dump_json accepts that model_dump does not (indent, ensure_ascii);
# every other option selects the same fields in both modes.
_JSON_ONLY_DUMP_OPTIONS: frozenset[str] = frozenset(
    inspect.signature(BaseModel.model_dump_json).parameters
) - frozenset(inspect.signature(BaseModel.model_dump).parameters)


MODEL_FIELD_REFUSALS: dict[str, RefusalSpec] = refusals(
    InvalidRequestError,
    {
        "model.field.range": "{model}.{field} value {value!r} violates its declared range: {constraint}",
        "model.field.nonfinite": "{model}.{field} value {value!r} must be finite",
        "model.field.type": "{model}.{field} expected {constraint}, got {value!r}",
        "model.field.missing": "{model}.{field} is required and was not supplied",
        "model.field.unknown": "{model} does not accept an unknown field {field!r}",
        "model.field.literal": "{model}.{field} value {value!r} is not one of {constraint}",
        "model.field.length": "{model}.{field} value {value!r} violates its declared length: {constraint}",
        "model.field.pattern": "{model}.{field} value {value!r} does not match {constraint}",
    },
)

_PYDANTIC_ERROR_TYPE_TO_CODE = {
    "greater_than": "model.field.range",
    "greater_than_equal": "model.field.range",
    "less_than": "model.field.range",
    "less_than_equal": "model.field.range",
    "multiple_of": "model.field.range",
    "finite_number": "model.field.nonfinite",
    "missing": "model.field.missing",
    "extra_forbidden": "model.field.unknown",
    "literal_error": "model.field.literal",
    "string_too_short": "model.field.length",
    "string_too_long": "model.field.length",
    "too_short": "model.field.length",
    "too_long": "model.field.length",
    "string_pattern_mismatch": "model.field.pattern",
}


def _field_error_code(pydantic_error_type: str) -> str | None:
    if pydantic_error_type in _PYDANTIC_ERROR_TYPE_TO_CODE:
        return _PYDANTIC_ERROR_TYPE_TO_CODE[pydantic_error_type]
    if pydantic_error_type.endswith(("_type", "_parsing")):
        return "model.field.type"
    return None


def _field_for_key(model_type: type[Any], key: object) -> tuple[str, FieldInfo] | None:
    """Resolve a pydantic field by Python name or validation alias."""
    if not isinstance(key, str):
        return None
    for name, field in getattr(model_type, "model_fields", {}).items():
        aliases = {name, field.alias}
        validation_alias = field.validation_alias
        choices = getattr(validation_alias, "choices", ())
        aliases.update(choice for choice in choices if isinstance(choice, str))
        if isinstance(validation_alias, str):
            aliases.add(validation_alias)
        if key in aliases:
            return name, field
    return None


_UNION_ORIGINS = (Union, UnionType)


def _unwrap(annotation: Any, discriminator: object) -> tuple[Any, object]:
    """Strip aliases, ``Annotated`` and ``Optional``, keeping any field-name discriminator."""
    while True:
        origin = get_origin(annotation)
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif origin is Annotated:
            annotation, *metadata = get_args(annotation)
            for item in metadata:
                if isinstance(item, FieldInfo) and item.discriminator is not None:
                    discriminator = item.discriminator
        elif origin in _UNION_ORIGINS:
            members = [arg for arg in get_args(annotation) if arg is not NoneType]
            if len(members) != 1:
                return annotation, discriminator
            annotation = members[0]
        else:
            return annotation, discriminator


def _union_members(annotation: Any) -> list[Any]:
    """Leaf members of a possibly nested union, with ``None`` dropped."""
    inner, _ = _unwrap(annotation, None)
    if get_origin(inner) in _UNION_ORIGINS:
        return [member for arg in get_args(inner) for member in _union_members(arg)]
    return [] if inner is NoneType else [inner]


def _literal_values(annotation: Any) -> tuple[Any, ...]:
    inner, _ = _unwrap(annotation, None)
    return get_args(inner) if get_origin(inner) is Literal else ()


def _tagged_member(annotation: Any, discriminator: str, tag: object) -> Any | None:
    """The single union member whose discriminator literal carries ``tag``."""
    matches = [
        member
        for member in _union_members(annotation)
        if isinstance(member, type)
        and issubclass(member, BaseModel)
        and (field := _field_for_key(member, discriminator)) is not None
        and tag in _literal_values(field[1].annotation)
    ]
    return matches[0] if len(matches) == 1 else None


def _walk(
    annotation: Any,
    discriminator: object,
    loc: tuple[Any, ...],
    index: int,
    owner: tuple[str, str] | None,
    error_type: str,
) -> tuple[str, str] | None:
    """Advance ``loc[index:]`` through ``annotation`` in lockstep with pydantic's
    validator tree; the innermost coded (model, field) owns the failure."""
    if index == len(loc):
        return owner
    annotation, discriminator = _unwrap(annotation, discriminator)
    segment = loc[index]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if not issubclass(annotation, CodedModel):
            return None
        resolved = _field_for_key(annotation, segment)
        if resolved is None:
            if error_type == "extra_forbidden" and index == len(loc) - 1:
                return annotation.__name__, str(segment)
            return None
        name, field = resolved
        owner = (annotation.__name__, name)
        return _walk(field.annotation, field.discriminator, loc, index + 1, owner, error_type)
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        # Untagged and callable-discriminated unions report every member's
        # failures under member labels; there is no single owner to name.
        if not isinstance(discriminator, str):
            return None
        member = _tagged_member(annotation, discriminator, segment)
        if member is None:
            return None
        return _walk(member, None, loc, index + 1, owner, error_type)
    if not isinstance(origin, type):
        return None
    args = get_args(annotation)
    if issubclass(origin, Mapping):
        if loc[index + 1 :] == ("[key]",):
            return owner
        value = args[1] if len(args) == 2 else Any
        return _walk(value, None, loc, index + 1, owner, error_type)
    if issubclass(origin, (Sequence, AbstractSet)) and isinstance(segment, int):
        if origin is tuple and args and args[-1] is not Ellipsis:
            if segment >= len(args):
                return None
            item = args[segment]
        else:
            item = args[0] if args else Any
        return _walk(item, None, loc, index + 1, owner, error_type)
    return None


def _resolve_field_error(
    model_type: type[Any], loc: tuple[Any, ...], error_type: str
) -> tuple[str, str] | None:
    """Resolve an error to its innermost CodedModel field, or leave it plain."""
    return _walk(model_type, None, loc, 0, None, error_type)


def _translate_field_errors(model_type: type[Any], exc: ValidationError) -> None:
    """Translate only when every pydantic error belongs to a coded field."""
    first: tuple[Mapping[str, Any], str, tuple[str, str]] | None = None
    for err in exc.errors():
        if (err.get("ctx") or {}).get("error") is not None:
            return
        code = _field_error_code(err["type"])
        target = _resolve_field_error(model_type, tuple(err.get("loc", ())), err["type"])
        if code is None or target is None:
            return
        first = first or (err, code, target)
    if first is None:
        return
    err, code, (model_name, field) = first
    ctx = err.get("ctx") or {}
    refuse(
        MODEL_FIELD_REFUSALS[code],
        model=model_name,
        field=field,
        loc=err.get("loc", ()),
        value=_safe_error_value(err.get("input")),
        constraint=ctx.get("le", ctx.get("ge", err.get("msg", ""))),
    )


def _preserve_base_model_init[InitT: Callable[..., Any]](func: InitT) -> InitT:
    update_wrapper(func, BaseModel.__init__)
    return func


class CodedValidationMixin:
    """Re-raise coded validator refusals from ordinary pydantic models."""

    @_preserve_base_model_init
    def __init__(self, /, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            unwrap_coded(exc)
            raise

    @classmethod
    def model_validate(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate(*args, **kwargs)  # ty: ignore[unresolved-attribute]
        except ValidationError as exc:
            unwrap_coded(exc)
            raise

    @classmethod
    def model_validate_json(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate_json(*args, **kwargs)  # ty: ignore[unresolved-attribute]
        except ValidationError as exc:
            unwrap_coded(exc)
            raise

    @classmethod
    def model_validate_strings(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate_strings(*args, **kwargs)  # ty: ignore[unresolved-attribute]
        except ValidationError as exc:
            unwrap_coded(exc)
            raise

    def model_dump(self, /, **kwargs: Any) -> dict[str, Any]:
        try:
            return super().model_dump(**kwargs)  # ty: ignore[unresolved-attribute]
        except PydanticSerializationError as exc:
            unwrap_coded_serialization(exc)
            raise

    def model_dump_json(self, /, **kwargs: Any) -> str:
        try:
            return super().model_dump_json(**kwargs)  # ty: ignore[unresolved-attribute]
        except PydanticSerializationError as exc:
            unwrap_coded_serialization(exc)
            # A JSON dump drops the chained cause; the python-mode dump over the same
            # selection raises the same serializer refusal with its cause attached.
            selection = {k: v for k, v in kwargs.items() if k not in _JSON_ONLY_DUMP_OPTIONS}
            try:
                super().model_dump(mode="json", **selection)  # ty: ignore[unresolved-attribute]
            except PydanticSerializationError as again:
                unwrap_coded_serialization(again)
            raise


class CodedModel(CodedValidationMixin):
    """Mixin that translates attributable pydantic field failures to coded errors.

    Direct construction and ``model_validate*`` calls surface
    ``InvalidRequestError`` for known field failures, while cross-field
    validators and unknown pydantic failure kinds remain ``ValidationError``.
    """

    @_preserve_base_model_init
    def __init__(self, /, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            unwrap_coded(exc)
            _translate_field_errors(type(self), exc)
            raise

    @classmethod
    def model_validate(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate(*args, **kwargs)
        except ValidationError as exc:
            unwrap_coded(exc)
            _translate_field_errors(cls, exc)
            raise

    @classmethod
    def model_validate_json(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate_json(*args, **kwargs)
        except ValidationError as exc:
            unwrap_coded(exc)
            _translate_field_errors(cls, exc)
            raise

    @classmethod
    def model_validate_strings(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate_strings(*args, **kwargs)
        except ValidationError as exc:
            unwrap_coded(exc)
            _translate_field_errors(cls, exc)
            raise


RETIRED_CODES: Mapping[str, str | tuple[str, ...] | None] = MappingProxyType(
    {
        "artifact.observational.governed_evidence_missing": None,
        "breakout.alpha": "estimation.diagnostics.alpha",
        "breakout.alpha_too_small": "estimation.meta.alpha_too_small",
        "breakout.segment_contrast_interval": "breakout.segment_contrast_absolute_interval",
        "contrast.assignment": None,
        "decision.absolute_decision.one_sided_nominal": "decision.relative_decision.one_sided_nominal",
        "decision.contrast_context.procedure_mapping_key": "decision.compiled_decision.procedure_mapping_key",
        "decision.contrast_decision.one_sided_alpha": "decision.arm_decision.one_sided_alpha",
        "definition.design.reject_bool": "definition.assignment.reject_bool",
        "definition.experiment.observation_schedule": (
            "definition.experiment.observation_end_before_end",
            "definition.experiment.observation_end_requires_end",
        ),
        "definition.method.finite_sample_cuped": "conversion_inference.finite_sample.cuped",
        "definition.validate_experiment.finite_sample_metric_type": "conversion_inference.finite_sample.metric_type",
        "estimation.absorption.alpha": "estimation.diagnostics.alpha",
        "estimation.adjust_aipw.aipw_control_mean": None,
        "estimation.adjust_dml.dml_control_mean": None,
        "estimation.adjust_iptw.iptw_hajek_control": None,
        "estimation.adjust_overlap.unit_ids_shape": "estimation.crossfit.fold_assignments_unit_ids_shape",
        "estimation.armstats.arm_stats.family_field_finite": "estimation.armstats.arm_stats.cross_field_finite",
        "estimation.armstats.arm_stats.var_d_needs_uptake_sum": "estimation.armstats.arm_stats.mean_d_needs_uptake_sum",
        "estimation.cate.alpha": "estimation.diagnostics.alpha",
        "estimation.contrast.contrast_partition.control_group_treatment": "estimation.contrast.contrast_stats.control_group_treatment",
        "estimation.contrast.contrast_partition.positive_cycles_contain_least_one": "estimation.contrast.contrast_partition.contain_least_one",
        "estimation.contrast.contrast_reduction_m2_finite": "estimation.contrast.contrast_reduction_delta_mean_finite",
        "estimation.crossfit.outer_split_unit_ids_shape": "estimation.crossfit.fold_assignments_unit_ids_shape",
        "estimation.encouragement.alpha": "estimation.diagnostics.alpha",
        "estimation.encouragement.mixture_priors_are": "estimation.adjust.prior.type",
        "estimation.encouragement.n_comparison_inference": None,
        "estimation.encouragement.nn_estimate_unknown_alternative": "estimation.binomial.unknown_alternative",
        "estimation.encouragement.one_sided_alpha": "estimation.inference.one_sided_alpha_doubles",
        "estimation.encouragement.prepare_encouragement_estimation_unknown_alternative": "estimation.binomial.unknown_alternative",
        "estimation.encouragement.retention": "readout.encouragement.retention",
        "estimation.engine.cluster.arm_needs_two": "estimation.encouragement.cluster.arm_needs_two",
        "estimation.engine.method.cuped_finite_sample": "conversion_inference.finite_sample.cuped",
        "estimation.family.e_bh_select_q_finite": "estimation.family.bh_select_q_finite",
        "estimation.inference.alpha_eff_underflows": "estimation.encouragement.alpha_eff_too",
        "estimation.inference.alpha_in_unit_interval": "estimation.diagnostics.alpha",
        "estimation.inference.combine_sequential_inference": None,
        "estimation.inference.dof_positive_for_t": "estimation.encouragement.dof_positive_reference",
        "estimation.inference.infer_ate_abs_diff_finite": "estimation.inference.infer_lift_abs_diff_finite",
        "estimation.inference.infer_ate_abs_se_finite": "estimation.inference.infer_lift_abs_se_finite",
        "estimation.inference.infer_ate_abs_se_positive": "estimation.inference.infer_lift_abs_se_positive",
        "estimation.inference.mixture_priors_are": "estimation.adjust.prior.type",
        "estimation.inference.n_comparison_inference": None,
        "estimation.inference.normal.expected_negative_part_scale": "estimation.priors.mixture_posterior.expected_negative_part_scale",
        "estimation.inference.normal.expected_positive_part_scale": "estimation.priors.mixture_posterior.expected_positive_part_scale",
        "estimation.inference.null_abs_finite": "estimation.inference.infer_ate_null_abs_finite",
        "estimation.inference.null_lift_finite": "estimation.inference.infer_ate_null_lift_finite",
        "estimation.inference.prior_sequential_inference": None,
        "estimation.inference.sequential_inference_cluster": None,
        "estimation.inference.unknown_alternative": "estimation.binomial.unknown_alternative",
        "estimation.meta.marginalized_segment.alpha_strictly_between": "estimation.meta.alpha_strictly_between",
        "estimation.meta.marginalized_segment.alpha_too_small": "estimation.meta.alpha_too_small",
        "estimation.meta.tau_prior_scale_overflow": None,
        "estimation.quantile.alpha": "estimation.diagnostics.alpha",
        "estimation.quantile.alpha_too_small": "estimation.meta.alpha_too_small",
        "estimation.results.estimate.lb_ub_both_when_lb_set": "estimation.results.estimate.lb_ub_both",
        "estimation.results.lift.p_value_null_lift_not_representable": "estimation.results.lift.p_value_null_lift_not_representable_dof",
        "estimation.results.lift.prob_favorable_cluster_robust_null_abs": "estimation.results.lift.p_value_cluster_robust_null_abs",
        "estimation.results.lift.prob_favorable_null_abs_missing_abs_se": "estimation.results.lift.p_value_null_abs_missing_abs_se",
        "estimation.sequential_inversion.inexact_value": "estimation.certified.inexact_value",
        "estimation.sequential_inversion.invalid_endpoint_reason": "estimation.sequential_likelihood.invalid_certificate_reason",
        "estimation.sequential_inversion.nonpositive_max_width": "estimation.certified.nonpositive_root_width",
        "estimation.sequential_inversion.state_prior_dimension_mismatch": "estimation.sequential_likelihood.state_prior_dimension_mismatch",
        "estimation.sequential_inversion.unknown_alternative": "estimation.sequential_likelihood.unknown_alternative",
        "estimation.sequential_likelihood.inexact_value": "estimation.certified.inexact_value",
        "estimation.sequential_likelihood.invalid_count": "estimation.certified.invalid_count",
        "estimation.sitewide.finite": "estimation.armstats.arm_stats.cross_field_finite",
        "estimation.targeting.adjustment_nonnumeric": "estimation.targeting.adjustment_dtype",
        "estimation.targeting.alpha": "estimation.diagnostics.alpha",
        "estimation.targeting.n_folds_least": "estimation.crossfit.n_folds_least",
        "estimation.targeting.outcome_nulls_non": "estimation.cate.outcome_nulls_non",
        "estimation.targeting.treatment_binary": "estimation.cate.treatment_binary",
        "estimation.variance.cluster_outcome_moments_needs_ref_den": "estimation.variance.ratio_moments_needs_ref_den",
        "estimation.variance.cluster_uptake_moments_nonpositive_size": "estimation.variance.cluster_outcome_moments_nonpositive_size",
        "estimation.winsor.bootstrap_replicate_failure": None,
        "estimation.winsor.bootstrap_tail_unresolved": None,
        "estimation.winsor.empty_region": None,
        "facade.analysis.source_context_design_disagrees_contrast": "facade.analysis.source_context_design_disagrees_arm",
        "frame.frame_panel.metric_was_declared": "frame.frame_totals.metric_was_declared",
        "frame.metric.finite_sample_metric_type": "conversion_inference.finite_sample.metric_type",
        "frame.moments.metric_covariate_column": "frame.validation.metric_covariate_impute_all_null",
        "plan.contrast_configs_template_mismatch": "plan.compile_configs_template_mismatch",
        "plan.contrast_configs_wrong_type": "plan.compile_configs_wrong_type",
        "power.alpha": "estimation.diagnostics.alpha",
        "power.segment_pairwise_minimum_n_per_arm_too_small": "power.segment_pairwise_achieved_n_per_arm_too_small",
        "power.sequential_power_enclosure_se_full_finite": "power.se_full_finite",
        "power.units_per_week": "power.units_per_week_finite_positive",
        "power.var_finite_strictly": "estimation.meta.var_finite_strictly",
        "query.builder.operation": (
            "query.builders.cluster_column_missing",
            "query.builders.cluster_size_imbalance",
            "query.builders.cluster_total_grain",
            "query.builders.cluster_undeclared",
            "query.builders.site_volume_metric_type",
        ),
        "query.builders.unit_totals_implemented": None,
        "query.builders.unknown_filter_op": None,
        "query.calendar.rolling_total_values_unknown_aggregation": "query.builders.unknown_aggregation",
        "query.calendar.total_period_values_unknown_aggregation": "query.builders.unknown_aggregation",
        "query.calendar.unknown_aggregation": "query.builders.unknown_aggregation",
        "query.calendar.unknown_grain": "query.calendar.period_start_unknown_grain",
        "query.fact_resolution.fact_source_property_overwrites_value_column": "query.fact_resolution.fact_source_property",
        "query.integrity.mixed_assignments": (
            "query.integrity.mixed_assignment_units",
            "query.integrity.unassigned_assignment_units",
        ),
        "query.session.warehouse_artifact.write_relation_sealed": "query.session.warehouse_artifact.publication_handle_sealed",
        "query.source.artifact_facade.verification_lazy_digest": "query.artifact_reader.artifact_moment.verification_lazy_digest",
        "readout.estimate_ate_every": "estimation.adjust.estimate_ate_every",
        "readout.inference.encouragement_run": None,
        "readout.unknown_estimand_supported": "estimation.encouragement.unknown_estimand_supported",
        "readout.value_scale_encouragement_absolute": "readout.encouragement.value_scale",
        "sequential.ratio.denominator_near_zero": None,
        "source.native.breakout_grain": "source.native.grain",
    }
)
"""Lookup only: maps a retired refusal code to its canonical replacement,
to a tuple of replacements when one code was split into several, or to
`None` if the code was deleted with no replacement (a genuinely dead
registration). Every retired/merged/deleted code in this codebase
gets exactly one entry here in the same change that removes it from the
live registry. A `CodedError` instance's own `.code` never changes after
construction -- this map is for callers who catch an old code and need
to migrate, and for `tests/test_retired_codes.py`'s enforcement."""


__all__ = [
    "CapabilityError",
    "CodedError",
    "CodedModel",
    "CodedValidationMixin",
    "DefinitionError",
    "IncrementDeprecationWarning",
    "IncrementRuntimeWarning",
    "IncrementWarning",
    "InvalidRequestError",
    "MODEL_FIELD_REFUSALS",
    "PACKAGE_FRAMES",
    "RETIRED_CODES",
    "RefusalSpec",
    "UnsupportedRequestError",
    "WarningSpec",
    "WireFormatError",
    "raiser",
    "refusals",
    "refuse",
    "unwrap_coded",
    "unwrap_coded_serialization",
    "warn",
]
