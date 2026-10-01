"""Lossless native identity canonicalization shared by sources and estimators."""

from __future__ import annotations

import math

import numpy as np

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.crossfit.identity_missing": "{what}[{index}] is missing (null/NaN); every row needs an identity",
        "estimation.crossfit.identity_collision": "{what} contains distinct native identities {first} and {second} that both render as {canonical!r} -- canonicalize to one representation before calling rather than let them silently merge into one group",
    },
)
_raise = raiser(_REFUSALS)


def canonical_id_strings(ids: np.ndarray, *, what: str) -> np.ndarray:
    """Injective canonical string per identity, never a randomized hash.

    Refuses a missing identity: ``None``, a Python/NumPy float NaN, a NumPy
    ``datetime64`` NaT, or a pandas ``NA``/``NaT`` scalar (matched by type
    name, so this module never imports pandas). Refuses when two *distinct*
    native values - unequal under ``!=``, regardless of type - render the
    same string, e.g. the int ``1`` and the str ``"1"`` both stringify to
    ``"1"``: merging those would silently fuse two different real-world
    identities into one group. A native value that renders identically to
    itself across a numpy/Python boundary (``np.str_('c1')`` vs ``'c1'``,
    ``np.int64(7)`` vs ``7``) is not a collision - it is the same value
    wearing a different dtype. The same value repeated across rows (a
    duplicated unit id, or a cluster id shared by its members) is not a
    collision either - it is exactly the broadcast this module relies on.
    """
    if ids.dtype.kind in "iu":
        # Integers can be neither missing nor colliding; decimal rendering is injective.
        return ids.astype(str).astype(object)
    if ids.dtype.kind == "U" or all(type(v) is str for v in ids):
        # Strings render as themselves, so neither refusal below can fire.
        return ids.astype(object)
    canon = np.empty(ids.shape[0], dtype=object)
    seen: dict[str, object] = {}
    for i in range(ids.shape[0]):
        v = ids[i]
        missing = (
            v is None
            or type(v).__name__ in {"NAType", "NaTType"}
            or (isinstance(v, (float, np.floating)) and math.isnan(v))
            or (isinstance(v, np.datetime64) and np.isnat(v))
        )
        if missing:
            _raise("estimation.crossfit.identity_missing", what=what, index=i)
        c = str(v)
        prior = seen.get(c)
        if prior is None:
            seen[c] = v
        elif prior != v:
            _raise(
                "estimation.crossfit.identity_collision",
                what=what,
                canonical=c,
                first=repr(prior),
                second=repr(v),
            )
        canon[i] = c
    return canon
