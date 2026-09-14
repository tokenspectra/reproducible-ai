"""Storage. A conversation is a trace: a root seed plus an ordered list of turns.

Turns hold the messages; steps hold the per-token evidence for reproducible
turns. Nothing here stores GPU state: a conversation can always be rebuilt
from (weights, envelope, prefix tokens, seed), which is what makes a resume
token small and portable.
"""

from __future__ import annotations

import secrets
import time
from typing import Any, ClassVar

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict] = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str] = mapped_column(String(300), default="New conversation")
    root_seed: Mapped[str] = mapped_column(String(64))
    seed_source: Mapped[str] = mapped_column(String(16), default="os_random")
    next_turn: Mapped[int] = mapped_column(Integer, default=0)
    forked_from: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    forked_at_turn: Mapped[int | None] = mapped_column(Integer, nullable=True)
    imported_from_token: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)

    turns: Mapped[list[Turn]] = relationship(
        back_populates="conversation", order_by="Turn.index", cascade="all, delete-orphan"
    )


class Turn(Base):
    __tablename__ = "turns"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), index=True
    )
    index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    engine: Mapped[str | None] = mapped_column(String(32), nullable=True)
    envelope_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sampling: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    evidence_kind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prev_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    hash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")
    steps: Mapped[list[Step]] = relationship(
        back_populates="turn", order_by="Step.position", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_turns_conv_index", "conversation_id", "index", unique=True),)

    @property
    def reproducible(self) -> bool:
        return self.engine == "reproducible"


class Step(Base):
    __tablename__ = "steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String(64), ForeignKey("turns.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    token_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)

    turn: Mapped[Turn] = relationship(back_populates="steps")


_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str, *, allow_idle_sleep: bool = True):
    """Create the engine.

    With allow_idle_sleep, pooled connections are recycled quickly and the pool
    is kept at zero when unused: an open database connection counts as outbound
    traffic to the host, which would stop an idle service from ever sleeping.
    """
    global _engine, _sessionmaker
    kwargs: dict = {"pool_pre_ping": True}
    if allow_idle_sleep and not database_url.startswith("sqlite"):
        kwargs |= {"pool_size": 1, "max_overflow": 4, "pool_recycle": 120, "pool_timeout": 30}
    _engine = create_async_engine(database_url, **kwargs)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def session() -> AsyncSession:
    if _sessionmaker is None:
        raise RuntimeError("database not initialised")
    return _sessionmaker()


async def create_all() -> None:
    """Create missing tables, then apply the small column additions since.

    No migration tool here on purpose: the schema is young and the additions so
    far are new nullable-with-default columns, which a couple of idempotent
    statements cover without dragging in Alembic for a demo.
    """
    if _engine is None:
        raise RuntimeError("database not initialised")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn)


async def _add_missing_columns(conn) -> None:
    from sqlalchemy import text

    dialect = conn.dialect.name
    if dialect == "postgresql":
        existing = set(
            (
                await conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'conversations'"
                    )
                )
            )
            .scalars()
            .all()
        )
    else:
        rows = (await conn.execute(text("PRAGMA table_info(conversations)"))).all()
        existing = {r[1] for r in rows}

    if "seq" not in existing:
        await conn.execute(
            text("ALTER TABLE conversations ADD COLUMN seq INTEGER NOT NULL DEFAULT 1")
        )
        # Backfill per user in creation order so existing rows get sensible numbers.
        if dialect == "postgresql":
            await conn.execute(
                text(
                    "UPDATE conversations c SET seq = n.rn FROM ("
                    "  SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) rn"
                    "  FROM conversations) n WHERE c.id = n.id"
                )
            )


async def dispose() -> None:
    if _engine is not None:
        await _engine.dispose()


async def next_sequence(db_session: AsyncSession, user_id: str) -> int:
    """The next per-user conversation number. Monotonic: deleting does not reuse."""
    from sqlalchemy import func

    current = (
        await db_session.execute(
            select(func.max(Conversation.seq)).where(Conversation.user_id == user_id)
        )
    ).scalar()
    return (current or 0) + 1


async def get_or_create_user(db: AsyncSession, email: str, name: str) -> User:
    found = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if found:
        if name and found.name != name:
            found.name = name
            await db.commit()
        return found
    user = User(id=new_id("usr"), email=email, name=name or email.split("@")[0])
    db.add(user)
    await db.commit()
    return user
