"""Identity.

The frontend owns Google sign-in and tells this service who the caller is with
a short-lived HMAC over (email, name, expiry) using a secret the two share.
That keeps Google's tokens on the frontend and keeps this service free of
OAuth, while still refusing anything it cannot verify.
"""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from overversion_api import db
from overversion_api.config import Settings

MAX_AGE_S = 600


def sign_identity(secret: str, email: str, name: str, expires_at: int) -> str:
    msg = f"{email}\n{name}\n{expires_at}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def make_header(secret: str, email: str, name: str, ttl_s: int = MAX_AGE_S) -> str:
    expires_at = int(time.time()) + ttl_s
    sig = sign_identity(secret, email, name, expires_at)
    return f"{email}|{name}|{expires_at}|{sig}"


async def current_user(
    request: Request,
    x_overversion_identity: str | None = Header(default=None),
) -> db.User:
    settings: Settings = request.app.state.settings
    if not settings.frontend_shared_secret:
        raise HTTPException(503, "identity verification is not configured")
    if not x_overversion_identity:
        raise HTTPException(401, "missing identity header")
    try:
        email, name, expires_raw, sig = x_overversion_identity.split("|", 3)
        expires_at = int(expires_raw)
    except ValueError:
        raise HTTPException(401, "malformed identity header") from None
    expected = sign_identity(settings.frontend_shared_secret, email, name, expires_at)
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(401, "bad identity signature")
    if expires_at < time.time():
        raise HTTPException(401, "identity expired")
    async with db.session() as s:
        return await db.get_or_create_user(s, email, name)


async def db_session() -> AsyncSession:
    async with db.session() as s:
        yield s


UserDep = Depends(current_user)
SessionDep = Depends(db_session)
