"""Length-prefixed SHA-256 draws for allocation protocol v1."""

from hashlib import sha256

_FULL = 10_000
_MAX_WORD = (1 << 32) - 1
_CUTOFF = ((_MAX_WORD + 1) // _FULL) * _FULL


def _hash_bytes(text: str) -> bytes | None:
    if len(text) > _MAX_WORD:
        return None
    encoded = text.encode("utf-8")
    return encoded if len(encoded) <= _MAX_WORD else None


def _prefix(stage: bytes, key: bytes) -> bytes:
    return b"".join(
        (
            b"increment.alloc.v1\0",
            len(stage).to_bytes(4, "big"),
            stage,
            len(key).to_bytes(4, "big"),
            key,
        )
    )


def _draw(prefix: bytes, unit: bytes) -> int:
    length = len(unit).to_bytes(4, "big")
    for attempt in range(_MAX_WORD + 1):
        digest = sha256(prefix)
        digest.update(length)
        digest.update(unit)
        digest.update(attempt.to_bytes(4, "big"))
        word = int.from_bytes(digest.digest()[:4], "big")
        if word < _CUTOFF:
            return word % _FULL
    raise AssertionError("Allocation draw exhausted its attempt counter.")
