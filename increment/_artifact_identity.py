"""Stdlib-only SHA-256 preimages for unit-day artifact format 1."""

from __future__ import annotations

import hashlib
from typing import Literal

ARTIFACT_V1_DOMAIN_ROOT: bytes = b"increment.unit-day-artifact\x00v1\x00"


def context_digest(canonical_json: str) -> str:
    return hashlib.sha256(
        ARTIFACT_V1_DOMAIN_ROOT + b"context\x00" + canonical_json.encode("utf-8")
    ).hexdigest()


def manifest_digest(canonical_body: bytes) -> str:
    return hashlib.sha256(ARTIFACT_V1_DOMAIN_ROOT + b"manifest\x00" + canonical_body).hexdigest()


def request_digest(canonical_body: bytes) -> str:
    return hashlib.sha256(ARTIFACT_V1_DOMAIN_ROOT + b"request\x00" + canonical_body).hexdigest()


def extension_digest(
    domain: Literal["extension-definition", "extension-source", "source-recipe"],
    kind: str,
    canonical_json: str,
) -> str:
    return hashlib.sha256(
        ARTIFACT_V1_DOMAIN_ROOT
        + domain.encode("ascii")
        + b"\x00"
        + kind.encode("utf-8")
        + b"\x00"
        + canonical_json.encode("utf-8")
    ).hexdigest()
