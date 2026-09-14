"""Seed derivation (DESIGN.md 6, ADR-0004).

One 256-bit root seed per session. Everything else is derived with BLAKE3's
key-derivation mode, which takes a context string and key material and is
designed for exactly this: independent subkeys that cannot be confused with
each other. Derivations are pure functions, so the API, the oracle and a
third-party verifier compute the same subkeys from the same root.
"""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass

import blake3

ROOT_SEED_BYTES = 32

_CTX_TURN = "reproducible-ai v1 turn seed"
_CTX_SAMPLING = "reproducible-ai v1 sampling seed"
_CTX_SR = "reproducible-ai v1 stochastic rounding seed"


@dataclass(frozen=True)
class RootSeed:
    value: bytes
    source: str  # "os_random" | "external"

    def __post_init__(self) -> None:
        if len(self.value) != ROOT_SEED_BYTES:
            raise ValueError(f"root seed must be {ROOT_SEED_BYTES} bytes, got {len(self.value)}")
        if self.source not in ("os_random", "external"):
            raise ValueError(f"unknown seed source {self.source!r}")

    @property
    def hex(self) -> str:
        return self.value.hex()

    @classmethod
    def from_hex(cls, hex_value: str, source: str = "external") -> RootSeed:
        return cls(bytes.fromhex(hex_value), source)


def new_root_seed() -> RootSeed:
    return RootSeed(secrets.token_bytes(ROOT_SEED_BYTES), "os_random")


def _derive(context: str, material: bytes) -> bytes:
    return blake3.blake3(material, derive_key_context=context).digest(length=32)


def derive_turn_seed(root: RootSeed, turn_index: int) -> bytes:
    if turn_index < 0:
        raise ValueError("turn_index must be >= 0")
    return _derive(_CTX_TURN, root.value + struct.pack("<Q", turn_index))


def derive_sampling_seed(turn_seed: bytes) -> bytes:
    return _derive(_CTX_SAMPLING, turn_seed)


def derive_sr_seed(turn_seed: bytes) -> bytes:
    return _derive(_CTX_SR, turn_seed)


def sampler_seed_u64(sampling_seed: bytes) -> int:
    """Engines such as SGLang take a 64-bit integer seed; use the first 8 bytes, little-endian.

    SGLang folds this with the absolute token position and the vocab index in
    a counter hash, so one integer per turn is enough for the batch-invariant
    property; the position is not ours to supply.
    """
    return struct.unpack("<Q", sampling_seed[:8])[0]
