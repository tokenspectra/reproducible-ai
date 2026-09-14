"""Resume tokens (DEMO.md, DESIGN.md 7).

A resume token is a recipe plus a commitment, not a snapshot: it names a trace
and a position in it, the envelope it was produced under, the root seed, a
commitment to the rendered prefix and the evidence hash of the last step. It
is signed with Ed25519 so the API can trust it without a database lookup and
so a third party can verify it with the public key, and it is transparent
(base64url of JSON) so anyone can read what it claims.

Wire format: ``rai1.<payload_b64url>.<signature_b64url>``.
"""

from __future__ import annotations

import base64
import json
from typing import Literal

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey
from pydantic import BaseModel, Field, ValidationError

PREFIX = "rai1"


class ResumeToken(BaseModel):
    v: Literal[1] = 1
    trace_id: str = Field(min_length=1, max_length=64)
    turn: int = Field(ge=0, description="index of the next turn to be generated")
    pos: int = Field(ge=0, description="absolute position of the last recorded token")
    model_id: str = Field(min_length=1, max_length=128)
    envelope_id: str = Field(min_length=1, max_length=64)
    root_seed: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix_hash: str = Field(pattern=r"^[0-9a-f]{32}$")
    evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    evidence_kind: str | None = Field(default=None, max_length=64)

    def payload_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class TokenSigner:
    def __init__(self, signing_key: SigningKey):
        self._key = signing_key

    @classmethod
    def generate(cls) -> TokenSigner:
        return cls(SigningKey.generate())

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> TokenSigner:
        return cls(SigningKey(bytes.fromhex(seed_hex)))

    @property
    def seed_hex(self) -> str:
        return bytes(self._key).hex()

    @property
    def verify_key_hex(self) -> str:
        return bytes(self._key.verify_key).hex()

    def sign(self, token: ResumeToken) -> str:
        payload = token.payload_bytes()
        sig = self._key.sign(payload).signature
        return f"{PREFIX}.{_b64e(payload)}.{_b64e(sig)}"


class TokenError(ValueError):
    pass


class TokenVerifier:
    def __init__(self, verify_key: VerifyKey):
        self._key = verify_key

    @classmethod
    def from_hex(cls, verify_key_hex: str) -> TokenVerifier:
        return cls(VerifyKey(bytes.fromhex(verify_key_hex)))

    def verify(self, wire: str) -> ResumeToken:
        parts = wire.strip().split(".")
        if len(parts) != 3 or parts[0] != PREFIX:
            raise TokenError("not a resume token")
        try:
            payload, sig = _b64d(parts[1]), _b64d(parts[2])
        except (ValueError, TypeError) as e:
            raise TokenError("malformed token encoding") from e
        try:
            self._key.verify(payload, sig)
        except BadSignatureError as e:
            raise TokenError("bad signature") from e
        try:
            token = ResumeToken.model_validate_json(payload)
        except ValidationError as e:
            raise TokenError(f"invalid payload: {e}") from e
        if token.payload_bytes() != payload:
            raise TokenError("payload not canonical")
        return token


def peek(wire: str) -> ResumeToken:
    """Decode without verifying. For display only; never trust the result."""
    parts = wire.strip().split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        raise TokenError("not a resume token")
    return ResumeToken.model_validate_json(_b64d(parts[1]))
