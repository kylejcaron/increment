from __future__ import annotations

import warnings
from types import MappingProxyType

import pytest

from increment.errors import (
    CapabilityError,
    CodedError,
    DefinitionError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    WireFormatError,
    refuse,
    warn,
)


def test_coded_error_requires_explicit_code_and_context() -> None:
    with pytest.raises(TypeError):
        CodedError("bad request")  # ty: ignore[missing-argument]

    original = {"request_id": "abc", "nested": {"items": [1]}}
    error = InvalidRequestError("bad request", code="request.invalid", context=original)

    original["request_id"] = "changed"
    original["new"] = True
    original["nested"]["items"].append(2)

    assert error.code == "request.invalid"
    assert str(error) == "bad request"
    assert error.context == MappingProxyType(
        {"request_id": "abc", "nested": MappingProxyType({"items": (1,)})}
    )
    assert not hasattr(error, "refusal_code")
    with pytest.raises(TypeError):
        error.context["other"] = "value"  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        error.context["nested"]["items"] += (2,)


def test_coded_error_code_and_context_attributes_are_read_only() -> None:
    error = CapabilityError("nope", code="cap.error", context={"metric": "rev"})
    with pytest.raises(AttributeError):
        error.code = "other.code"  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        error.context = {"metric": "aov"}  # ty: ignore[invalid-assignment]
    assert error.code == "cap.error"
    assert error.context == MappingProxyType({"metric": "rev"})


def test_definition_error_derives_metadata_from_context() -> None:
    sources = {"metric:m": ["a.yaml", "b.yaml"]}
    context = {"path": "defs", "sources": sources}
    error = DefinitionError("duplicate", code="definition.duplicates", context=context)

    sources["metric:m"].append("changed.yaml")
    sources["new"] = ["new.yaml"]

    assert error.path == "defs"
    assert error.sources == MappingProxyType({"metric:m": ("a.yaml", "b.yaml")})
    assert error.context["path"] == "defs"
    assert error.context["sources"] == MappingProxyType({"metric:m": ("a.yaml", "b.yaml")})
    assert error.code == "definition.duplicates"
    assert error.message == "duplicate"
    with pytest.raises(TypeError):
        error.sources["metric:m"] += ("changed.yaml",)


def test_refuse_raises_declared_subclass_with_code_and_context() -> None:
    def render(**context: object) -> str:
        return f"missing {context['name']}"

    spec = RefusalSpec("source.missing", CapabilityError, render)
    with pytest.raises(CapabilityError) as raised:
        refuse(spec, name="metric")

    assert raised.value.code == "source.missing"
    assert raised.value.context == {"name": "metric"}


def test_unsupported_request_is_coded_and_keeps_not_implemented_catch() -> None:
    spec = RefusalSpec(
        "request.unsupported",
        UnsupportedRequestError,
        lambda **_: "not supported",
    )

    with pytest.raises(NotImplementedError) as raised:
        refuse(spec)

    assert isinstance(raised.value, UnsupportedRequestError)
    assert isinstance(raised.value, CodedError)
    assert raised.value.code == "request.unsupported"


def test_direct_error_calls_preserve_value_error_behavior() -> None:
    errors = [
        CapabilityError("failure", code="capability.test", context={}),
        InvalidRequestError("failure", code="request.test", context={}),
        WireFormatError("failure", code="wire.test", context={}),
        DefinitionError("failure", code="definition.test", context={}),
        UnsupportedRequestError("failure", code="unsupported.test", context={}),
    ]
    for error in errors:
        with pytest.raises(ValueError):
            raise error


def test_each_coded_subclass_uses_the_explicit_constructor_contract() -> None:
    classes = (
        CapabilityError,
        InvalidRequestError,
        WireFormatError,
        DefinitionError,
        UnsupportedRequestError,
    )
    for error_type in classes:
        error = error_type("failure", code="test.failure", context={"owner": "test"})
        assert error.code == "test.failure"
        assert error.context["owner"] == "test"


def test_refuse_with_a_mistyped_kwarg_fails_at_call_time() -> None:
    from increment.errors import InvalidRequestError, RefusalSpec, refuse

    spec = RefusalSpec("test.mistyped", InvalidRequestError, lambda *, path: f"path={path}")
    with pytest.raises(TypeError):
        refuse(spec, pth="/tmp/x")


def test_refuse_with_a_template_spec_missing_a_declared_key_raises_assertion_error() -> None:
    from increment.errors import InvalidRequestError, RefusalSpec, refuse

    spec = RefusalSpec("test.template_missing_key", InvalidRequestError, template="path={path}")
    with pytest.raises(AssertionError):
        refuse(spec, pth="/tmp/x")


def test_refusals_builder_accepts_template_pair_and_prebuilt_spec() -> None:
    from increment.errors import InvalidRequestError, RefusalSpec, refusals

    prebuilt = RefusalSpec("test.prebuilt", InvalidRequestError, lambda *, x: str(x))
    registry = refusals(
        InvalidRequestError,
        {
            "test.plain_template": "value is {value}",
            "test.template_with_extra_key": ("a constant message", ("alpha",)),
            "test.prebuilt": prebuilt,
        },
    )
    assert registry["test.plain_template"].keys == frozenset({"value"})
    assert registry["test.template_with_extra_key"].keys == frozenset({"alpha"})
    assert registry["test.prebuilt"] is prebuilt


def test_raiser_builds_a_working_raise_function() -> None:
    from increment.errors import InvalidRequestError, raiser, refusals

    registry = refusals(InvalidRequestError, {"test.raiser_demo": "got {value}"})
    _raise = raiser(registry)
    with pytest.raises(InvalidRequestError) as raised:
        _raise("test.raiser_demo", value=3)
    assert raised.value.code == "test.raiser_demo"
    assert str(raised.value) == "got 3"


def test_refuse_with_a_template_spec_accepts_a_self_keyed_context() -> None:
    from increment.errors import InvalidRequestError, RefusalSpec, refuse

    spec = RefusalSpec("test.self_key", InvalidRequestError, template="value is {self!r}")
    with pytest.raises(InvalidRequestError) as raised:
        refuse(spec, self="x")
    assert raised.value.code == "test.self_key"
    assert str(raised.value) == "value is 'x'"


def _pickleable_render(*, x: object) -> str:
    return str(x)


def test_refusal_spec_template_variant_survives_pickle_and_deepcopy_round_trip() -> None:
    import copy
    import pickle

    spec = RefusalSpec("test.pickle_template", InvalidRequestError, template="x={x}")
    for restored in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec)):
        assert restored == spec
        assert restored.code == spec.code
        assert restored.keys == spec.keys
        assert restored.render(x=3) == "x=3"


def test_refusal_spec_template_variant_supports_dataclasses_replace() -> None:
    import dataclasses

    spec = RefusalSpec("test.replace_template", InvalidRequestError, template="x={x}")
    replaced = dataclasses.replace(spec, code="test.replace_template.renamed")
    assert replaced.code == "test.replace_template.renamed"
    assert replaced.keys == spec.keys
    assert replaced.render(x=3) == "x=3"
    with pytest.raises(AssertionError) as raised:
        replaced.render()
    assert "test.replace_template.renamed" in str(raised.value)


def test_refusal_spec_render_variant_survives_pickle_and_deepcopy_round_trip() -> None:
    import copy
    import pickle

    spec = RefusalSpec("test.pickle_render", InvalidRequestError, _pickleable_render)
    for restored in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec)):
        assert restored == spec
        assert restored.code == spec.code
        assert restored.render(x=3) == "3"


def test_copy_sources_accepts_a_tuple_of_strings() -> None:
    from increment.errors import DefinitionError

    err = DefinitionError(
        "boom", code="test.tuple_sources", context={"sources": {"a": ("x.yaml",)}}
    )
    assert err.sources == {"a": ("x.yaml",)}


def test_refusal_spec_post_init_rejects_positional_only_renderer_via_assertion() -> None:
    def _bad_renderer(x: object, /) -> str:
        return str(x)

    with pytest.raises(AssertionError):
        RefusalSpec("test.bad_renderer", InvalidRequestError, _bad_renderer)


def test_warning_spec_post_init_rejects_positional_only_renderer_via_assertion() -> None:
    def _bad_renderer(x: object, /) -> str:
        return str(x)

    with pytest.raises(AssertionError):
        WarningSpec("test.bad_warning_renderer", IncrementWarning, _bad_renderer)


def test_context_snapshots_array_values_instead_of_aliasing_them() -> None:
    np = pytest.importorskip("numpy")
    arr = np.array([1.0, 2.0])
    err = InvalidRequestError(
        "m", code="test.array", context={"arr": arr, "scalar": np.float64(3.0)}
    )
    arr[0] = 99.0
    assert err.context["arr"] == (1.0, 2.0)
    assert err.context["scalar"] == 3.0
    assert err.context == {"arr": (1.0, 2.0), "scalar": 3.0}


def test_context_leaves_non_numpy_objects_with_tolist_untouched() -> None:
    class _Fake:
        def tolist(self, required):
            raise AssertionError("must not be called")

    fake = _Fake()
    err = InvalidRequestError("m", code="test.fake", context={"obj": fake})
    assert err.context["obj"] is fake


def test_coded_error_survives_pickle_and_deepcopy_round_trip() -> None:
    import copy
    import pickle

    original = {"request_id": "abc", "nested": {"items": [1, 2]}}
    error = InvalidRequestError("bad request", code="request.invalid", context=original)

    restored = pickle.loads(pickle.dumps(error))
    assert restored.code == error.code
    assert restored.context == error.context
    assert str(restored) == str(error)
    assert isinstance(restored, InvalidRequestError)
    with pytest.raises(TypeError):
        restored.context["nested"]["items"] += (3,)  # still frozen

    cloned = copy.deepcopy(error)
    assert cloned.code == error.code
    assert cloned.context == error.context
    assert cloned is not error
    assert cloned.context is not error.context


def test_coded_warning_survives_pickle_and_deepcopy_round_trip() -> None:
    import copy
    import pickle

    from increment.errors import IncrementDeprecationWarning, IncrementRuntimeWarning

    original = {"request_id": "abc", "nested": {"items": [1, 2]}}
    for cls in (IncrementWarning, IncrementRuntimeWarning, IncrementDeprecationWarning):
        warning = cls("bad state", code=f"test.{cls.__name__.lower()}", context=original)

        restored = pickle.loads(pickle.dumps(warning))
        assert restored.code == warning.code
        assert restored.context == warning.context
        assert str(restored) == str(warning)
        assert type(restored) is cls
        with pytest.raises(TypeError):
            restored.context["nested"]["items"] += (3,)  # still frozen

        cloned = copy.deepcopy(warning)
        assert cloned.code == warning.code
        assert cloned.context == warning.context
        assert cloned is not warning
        assert cloned.context is not warning.context


def test_definition_error_pickle_round_trip_preserves_derived_attributes() -> None:
    import pickle

    sources = {"metric:m": ["a.yaml", "b.yaml"]}
    error = DefinitionError(
        "duplicate", code="definition.duplicates", context={"path": "defs", "sources": sources}
    )
    restored = pickle.loads(pickle.dumps(error))
    assert restored.path == error.path
    assert restored.sources == error.sources
    assert restored.message == error.message
    assert restored.code == error.code
    with pytest.raises(TypeError):
        restored.sources["metric:m"] += ("x",)  # still frozen


def _all_coded_error_subclasses() -> list[type]:
    """Walk the live class tree instead of hand-listing subclasses, so a
    new CodedError subclass is covered automatically."""
    seen: set[type] = set()
    frontier = [CodedError]
    while frontier:
        cls = frontier.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                frontier.append(sub)
    return sorted(seen, key=lambda c: c.__qualname__)


def test_every_coded_error_subclass_survives_pickle_and_deepcopy() -> None:
    import copy
    import importlib
    import pickle

    # Import every module that defines a CodedError subclass so the walk
    # below sees the full live class tree, not just what test_errors.py's
    # own imports happen to have loaded.
    for module in (
        "increment.errors",
        "increment.estimation.binomial_rr",
        "increment.estimation._adjust.overlap",
        "increment.query.artifact_contract",
        "increment.query.artifact_digest",
        "increment.estimation.inference",
        "increment.estimation.engine",
    ):
        importlib.import_module(module)
    from increment.estimation.engine import _NonPositiveMeanFailure
    from increment.estimation.inference import LiftGuardError

    # Subclasses with their own constructor signature are built through it.
    factories = {
        LiftGuardError: lambda: LiftGuardError("probe message", reason="zero_variance"),
        _NonPositiveMeanFailure: lambda: _NonPositiveMeanFailure(
            "probe message", abs_diff=1.0, abs_se_t=0.5, abs_se_c=0.5
        ),
    }

    subclasses = _all_coded_error_subclasses()
    assert len(subclasses) >= 7, "expected at least the known CodedError subclasses"

    for cls in subclasses:
        factory = factories.get(cls)
        error = (
            factory()
            if factory is not None
            else cls("probe message", code=f"test.{cls.__name__.lower()}", context={"k": (1, 2)})
        )
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is cls
        assert restored.code == error.code
        assert restored.context == error.context
        assert str(restored) == str(error)

        cloned = copy.deepcopy(error)
        assert type(cloned) is cls
        assert cloned.code == error.code
        assert cloned.context == error.context


def test_warn_attributes_a_direct_call_to_its_own_caller() -> None:
    """`warn()` adds its own frame internally, so a direct call with
    `stacklevel=1` (the default `warnings.warn` meaning: attribute to
    my own caller) must land on the `warn(...)` call line, not inside
    `errors.py`."""
    spec = WarningSpec("test.warn.direct_attribution", IncrementWarning, lambda **_: "direct")

    def _caller_site() -> None:
        warn(spec, stacklevel=1)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _caller_site()
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert rec[0].lineno == _caller_site.__code__.co_firstlineno + 1


def test_warn_helper_wrapper_absorbs_its_own_frame() -> None:
    """A module-local `_warn(code, /, *, stacklevel=2, **context)` helper
    that forwards `stacklevel=stacklevel + 1` to `warn()` must attribute
    the warning to the exact frame a literal `warnings.warn(msg,
    category, stacklevel=2)` written at `_wrapped_site`'s call line
    would have -- i.e. `_wrapped_site`'s own caller, not the helper's or
    `warn()`'s frame, and not `_wrapped_site` itself."""
    _warnings: dict[str, WarningSpec] = {}
    _warnings["test.warn.helper_attribution"] = WarningSpec(
        "test.warn.helper_attribution", IncrementWarning, lambda **_: "via helper"
    )

    def _warn(code: str, /, *, stacklevel: int = 2) -> None:
        warn(_warnings[code], stacklevel=stacklevel + 1)

    def _wrapped_site() -> None:
        _warn("test.warn.helper_attribution", stacklevel=2)

    def _outer_caller() -> None:
        _wrapped_site()

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _outer_caller()
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert rec[0].lineno == _outer_caller.__code__.co_firstlineno + 1


def test_warn_skip_file_prefixes_excludes_its_own_frame() -> None:
    """`skip_file_prefixes` must also skip `warn()`'s own frame in
    `errors.py`, exactly like the `stacklevel` path does -- a caller
    passing only ITS OWN prefix must not have the warning misattributed
    to `errors.py`."""
    spec = WarningSpec(
        "test.warn.skip_prefix_attribution", IncrementWarning, lambda **_: "via prefix"
    )

    def _caller_site() -> None:
        warn(spec, skip_file_prefixes=("/definitely-not-a-real-prefix-xyz",))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _caller_site()
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert rec[0].lineno == _caller_site.__code__.co_firstlineno + 1


def test_increment_runtime_warning_is_caught_by_runtime_warning_filter() -> None:
    """`IncrementRuntimeWarning` must satisfy `filterwarnings(category=
    RuntimeWarning)` -- the whole point of the multiply-inherited category."""
    from increment.errors import IncrementRuntimeWarning

    spec = WarningSpec("test.warn.runtime_category", IncrementRuntimeWarning, lambda **_: "runtime")
    with pytest.warns(RuntimeWarning) as rec:
        warn(spec)
    message = rec[0].message
    assert isinstance(message, IncrementRuntimeWarning)
    assert message.code == "test.warn.runtime_category"
