"""Canonical hashing (DESIGN.md 5, ADR-0009).

Every hash here is BLAKE3 over a byte string with a fixed layout, truncated to
128 bits and rendered as 32 lowercase hex characters. Layouts are versioned by
a domain prefix so that two different kinds of evidence can never collide by
accident and so a layout change is a visible format change.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterable, Sequence
from typing import Any

import blake3

HASH_BYTES = 16  # 128 bits

_DOMAIN_TOKENS = b"rai:prefix-tokens:v1\0"
_DOMAIN_TOPK = b"rai:top-logprobs:v1\0"
_DOMAIN_LOGITS = b"rai:fp32-logits:v1\0"
_DOMAIN_EVENT = b"rai:event:v1\0"


def blake3_128(data: bytes) -> str:
    """Hex digest of BLAKE3(data) truncated to 128 bits."""
    return blake3.blake3(data).digest(length=HASH_BYTES).hex()


def hash_prefix_tokens(token_ids: Iterable[int]) -> str:
    """Commitment to a rendered token sequence: little-endian uint32 per id."""
    h = blake3.blake3(_DOMAIN_TOKENS)
    for tid in token_ids:
        if tid < 0 or tid > 0xFFFFFFFF:
            raise ValueError(f"token id out of uint32 range: {tid}")
        h.update(struct.pack("<I", tid))
    return h.digest(length=HASH_BYTES).hex()


def hash_top_logprobs(pairs: Sequence[tuple[int, float]], k: int = 32) -> str:
    """Evidence hash for engines that expose top-k logprobs rather than raw logits.

    Canonical form: the first ``k`` (token_id, logprob) pairs sorted by
    logprob descending and token id ascending as the tie-break, each encoded
    as little-endian uint32 + float32. Sorting here means the hash does not
    depend on the order the engine happened to return them; the float32 cast
    is deliberate because engines return float32-derived values as JSON
    doubles and re-serialisation must not change the hash.
    """
    canon = sorted(pairs, key=lambda p: (-p[1], p[0]))[:k]
    h = blake3.blake3(_DOMAIN_TOPK)
    h.update(struct.pack("<I", len(canon)))
    for tid, lp in canon:
        h.update(struct.pack("<If", tid, lp))
    return h.digest(length=HASH_BYTES).hex()


def hash_fp32_logits(logits_le_bytes: bytes) -> str:
    """Evidence hash for the oracle: the full fp32 logits vector, little-endian, vocab order."""
    if len(logits_le_bytes) % 4:
        raise ValueError("fp32 logits byte length must be a multiple of 4")
    h = blake3.blake3(_DOMAIN_LOGITS)
    h.update(struct.pack("<I", len(logits_le_bytes) // 4))
    h.update(logits_le_bytes)
    return h.digest(length=HASH_BYTES).hex()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def hash_event(event: dict[str, Any], prev_hash: str | None) -> str:
    """Hash of one trace event for the chain: BLAKE3(domain || prev || canonical json without 'hash')."""
    body = {k: v for k, v in event.items() if k != "hash"}
    h = blake3.blake3(_DOMAIN_EVENT)
    h.update((prev_hash or "").encode("ascii"))
    h.update(b"\0")
    h.update(canonical_json(body))
    return h.digest(length=HASH_BYTES).hex()
