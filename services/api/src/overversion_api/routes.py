"""Conversation routes: list, create, fork, send, and resume tokens."""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from reproducible_ai.hashing import hash_event, hash_prefix_tokens
from reproducible_ai.seeds import (
    RootSeed,
    derive_sampling_seed,
    derive_turn_seed,
    new_root_seed,
    sampler_seed_u64,
)
from reproducible_ai.token import ResumeToken, TokenError, TokenSigner, TokenVerifier
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from overversion_api import db
from overversion_api.auth import current_user, db_session
from overversion_api.engines import EngineId, EngineRegistry, SamplingParams
from overversion_api.status import StatusTracker

router = APIRouter(prefix="/v1")

UserT = Annotated[db.User, Depends(current_user)]
SessionT = Annotated[AsyncSession, Depends(db_session)]


# ---------- wire models ----------


class CreateConversation(BaseModel):
    title: str | None = None
    seed_hex: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    resume_token: str | None = None
    token_mode: Literal["seed", "state"] = Field(
        default="seed",
        description=(
            "What to take from the token. 'seed' starts an empty conversation on the same root "
            "seed, so the same first message produces the same answer as it did there. 'state' "
            "copies the whole history and continues from where the token was issued."
        ),
    )


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=32000)
    engine: EngineId = EngineId.REPRODUCIBLE
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    thinking: bool = False


# ---------- helpers ----------


def signer(request: Request) -> TokenSigner:
    seed = request.app.state.settings.token_signing_seed_hex
    if not seed:
        raise HTTPException(503, "token signing is not configured")
    return TokenSigner.from_seed_hex(seed)


async def load_conversation(
    s: AsyncSession, conversation_id: str, user: db.User, *, owned: bool = True
) -> db.Conversation:
    q = (
        select(db.Conversation)
        .where(db.Conversation.id == conversation_id)
        .options(selectinload(db.Conversation.turns).selectinload(db.Turn.steps))
    )
    conv = (await s.execute(q)).scalar_one_or_none()
    if conv is None:
        raise HTTPException(404, "conversation not found")
    if owned and conv.user_id != user.id:
        raise HTTPException(403, "not your conversation")
    return conv


def conversation_json(
    conv: db.Conversation, *, with_turns: bool = False, source_title: str | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": conv.id,
        "seq": conv.seq,
        "title": conv.title,
        "next_turn": conv.next_turn,
        "seed_source": conv.seed_source,
        "forked_from": conv.forked_from,
        "forked_at_turn": conv.forked_at_turn,
        "imported_from_token": conv.imported_from_token,
        "forked_from_title": source_title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
    }
    if with_turns:
        out["turns"] = [
            {
                "index": t.index,
                "role": t.role,
                "content": t.content,
                "engine": t.engine,
                "envelope_id": t.envelope_id,
                "seed": t.seed,
                "finish_reason": t.finish_reason,
                "evidence_kind": t.evidence_kind,
                "steps": len(t.steps),
                "hash": t.hash,
            }
            for t in conv.turns
        ]
    return out


def messages_for(conv: db.Conversation) -> list[dict[str, str]]:
    return [{"role": t.role, "content": t.content} for t in conv.turns]


def prefix_commitment(conv: db.Conversation) -> str:
    """Commitment to the conversation prefix.

    The real thing hashes rendered token ids; until the tokenizer lives on this
    side, hash the message bytes in the same canonical way so the field is
    populated, stable and order-sensitive. The kind is recorded on the token so
    a verifier knows which it is.
    """
    ids: list[int] = []
    for t in conv.turns:
        ids.extend(f"{t.role}:{t.content}".encode()[:4096])
    return hash_prefix_tokens(ids)


# ---------- routes ----------


@router.get("/conversations")
async def list_conversations(user: UserT, s: SessionT) -> dict[str, Any]:
    q = (
        select(db.Conversation)
        .where(db.Conversation.user_id == user.id)
        .order_by(db.Conversation.updated_at.desc())
        .options(selectinload(db.Conversation.turns).selectinload(db.Turn.steps))
    )
    rows = (await s.execute(q)).scalars().all()
    return {"conversations": [conversation_json(c) for c in rows]}


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: CreateConversation, request: Request, user: UserT, s: SessionT
) -> dict[str, Any]:
    conv = db.Conversation(
        id=db.new_id("cnv"),
        user_id=user.id,
        seq=await db.next_sequence(s, user.id),
        title=body.title or "New conversation",
        root_seed=new_root_seed().hex,
        seed_source="os_random",
    )

    source_title: str | None = None
    if body.resume_token:
        token = _verify_token(request, body.resume_token)
        conv.root_seed = token.root_seed
        conv.seed_source = "external"
        conv.imported_from_token = True

        if body.token_mode == "seed":
            # Nothing is copied. An empty conversation on the same root seed, so
            # the same first message draws the same subkey and returns the same
            # answer as it did in the conversation the token came from.
            conv.title = body.title or "New conversation"
            s.add(conv)
        else:
            source = (
                await s.execute(
                    select(db.Conversation)
                    .where(db.Conversation.id == token.trace_id)
                    .options(selectinload(db.Conversation.turns).selectinload(db.Turn.steps))
                )
            ).scalar_one_or_none()
            if source is None:
                raise HTTPException(404, "the token's conversation no longer exists")
            if prefix_commitment(source) != token.prefix_hash:
                raise HTTPException(
                    409, "the token's conversation has changed since it was issued"
                )
            conv.next_turn = token.turn
            conv.forked_from = source.id
            conv.forked_at_turn = token.turn
            conv.title = body.title or source.title
            s.add(conv)
            await s.flush()
            for t in source.turns:
                s.add(_copy_turn(t, conv.id))
            source_title = source.title
    elif body.seed_hex:
        conv.root_seed = RootSeed.from_hex(body.seed_hex).hex
        conv.seed_source = "external"
        s.add(conv)
    else:
        s.add(conv)

    await s.commit()
    conv = await load_conversation(s, conv.id, user)
    return conversation_json(conv, with_turns=True, source_title=source_title)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user: UserT, s: SessionT) -> dict[str, Any]:
    conv = await load_conversation(s, conversation_id, user)
    return conversation_json(conv, with_turns=True)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, user: UserT, s: SessionT) -> None:
    conv = await load_conversation(s, conversation_id, user)
    await s.delete(conv)
    await s.commit()


@router.post("/conversations/{conversation_id}/fork", status_code=201)
async def fork_conversation(conversation_id: str, user: UserT, s: SessionT) -> dict[str, Any]:
    source = await load_conversation(s, conversation_id, user)
    fork = db.Conversation(
        id=db.new_id("cnv"),
        user_id=user.id,
        seq=await db.next_sequence(s, user.id),
        title=f"Fork of {source.title}",
        root_seed=source.root_seed,
        seed_source=source.seed_source,
        next_turn=source.next_turn,
        forked_from=source.id,
        forked_at_turn=source.next_turn,
    )
    s.add(fork)
    await s.flush()
    source_title = source.title
    for t in source.turns:
        s.add(_copy_turn(t, fork.id))
    await s.commit()
    fork = await load_conversation(s, fork.id, user)
    return conversation_json(fork, with_turns=True, source_title=source_title)


def _copy_turn(t: db.Turn, conversation_id: str) -> db.Turn:
    return db.Turn(
        id=db.new_id("trn"),
        conversation_id=conversation_id,
        index=t.index,
        role=t.role,
        content=t.content,
        engine=t.engine,
        envelope_id=t.envelope_id,
        seed=t.seed,
        sampling=t.sampling,
        finish_reason=t.finish_reason,
        evidence_kind=t.evidence_kind,
        prev_hash=t.prev_hash,
        hash=t.hash,
    )


@router.get("/conversations/{conversation_id}/token")
async def issue_token(
    conversation_id: str, request: Request, user: UserT, s: SessionT
) -> dict[str, Any]:
    conv = await load_conversation(s, conversation_id, user)
    repro = [t for t in conv.turns if t.reproducible]
    if not repro:
        raise HTTPException(
            409, "this conversation has no reproducible turn yet, so there is no state to resume"
        )
    last = repro[-1]
    steps = last.steps
    token = ResumeToken(
        trace_id=conv.id,
        turn=conv.next_turn,
        pos=steps[-1].position if steps else 0,
        model_id=request.app.state.settings.infer_model,
        envelope_id=last.envelope_id or "unknown",
        root_seed=conv.root_seed,
        prefix_hash=prefix_commitment(conv),
        evidence_hash=steps[-1].evidence_hash if steps else None,
        evidence_kind=last.evidence_kind,
    )
    return {"token": signer(request).sign(token), "payload": token.model_dump(exclude_none=True)}


@router.post("/tokens/inspect")
async def inspect_token(body: dict[str, str], request: Request, user: UserT) -> dict[str, Any]:
    token = _verify_token(request, body.get("token", ""))
    return {"valid": True, "payload": token.model_dump(exclude_none=True)}


def _verify_token(request: Request, wire: str) -> ResumeToken:
    seed = request.app.state.settings.token_signing_seed_hex
    if not seed:
        raise HTTPException(503, "token signing is not configured")
    verifier = TokenVerifier.from_hex(TokenSigner.from_seed_hex(seed).verify_key_hex)
    try:
        return verifier.verify(wire)
    except TokenError as e:
        raise HTTPException(400, f"invalid resume token: {e}") from e


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str, body: SendMessage, request: Request, user: UserT, s: SessionT
):
    conv = await load_conversation(s, conversation_id, user)
    registry: EngineRegistry = request.app.state.registry
    tracker: StatusTracker = request.app.state.status
    status = (await tracker.current()).get(body.engine)
    if status is None or not status.up:
        raise HTTPException(
            409,
            f"{body.engine} is not available: {status.detail if status else 'unknown engine'}",
        )
    engine = registry[body.engine]

    # Build the history from what we already hold plus the message being sent.
    # Reloading here would return the identity-mapped conversation whose turn
    # collection predates this insert, which silently sends an empty history.
    history = [*messages_for(conv), {"role": "user", "content": body.content}]

    user_turn = db.Turn(
        id=db.new_id("trn"),
        conversation_id=conv.id,
        index=conv.next_turn,
        role="user",
        content=body.content,
    )
    _chain(conv, user_turn)
    s.add(user_turn)
    conv.next_turn += 1
    conv.updated_at = time.time()
    if conv.title == "New conversation":
        conv.title = body.content[:60] + ("..." if len(body.content) > 60 else "")
    await s.commit()

    turn_seed = derive_turn_seed(RootSeed.from_hex(conv.root_seed), conv.next_turn)
    seed_u64 = sampler_seed_u64(derive_sampling_seed(turn_seed)) % (2**31)
    params = SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        seed=seed_u64 if engine.capabilities.accepts_seed else None,
    )
    title = conv.title
    next_turn = conv.next_turn

    async def stream():
        yield _sse("start", {"turn": next_turn, "engine": str(body.engine), "title": title})
        text_parts: list[str] = []
        steps: list[tuple[int, int | None, str | None]] = []
        kind: str | None = None
        finish: str | None = None
        try:
            async for chunk in engine.stream(history, params):
                if chunk.text:
                    text_parts.append(chunk.text)
                    yield _sse("delta", {"text": chunk.text})
                for st in chunk.steps:
                    steps.append((st.position, st.token_id, st.evidence_hash))
                    kind = kind or st.evidence_kind
                if chunk.finish_reason:
                    finish = chunk.finish_reason
        except Exception as e:  # noqa: BLE001 - any upstream failure must reach the client
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"[:300]})
            return

        async with db.session() as s2:
            conv2 = (
                await s2.execute(
                    select(db.Conversation).where(db.Conversation.id == conversation_id)
                )
            ).scalar_one()
            turn = db.Turn(
                id=db.new_id("trn"),
                conversation_id=conversation_id,
                index=conv2.next_turn,
                role="assistant",
                content="".join(text_parts),
                engine=str(body.engine),
                envelope_id=status.envelope_id,
                seed=params.seed,
                sampling={
                    "temperature": params.temperature,
                    "top_p": params.top_p,
                    "max_tokens": params.max_tokens,
                },
                finish_reason=finish,
                evidence_kind=kind,
                prev_hash=None,
            )
            turn.hash = hash_event(
                {"index": turn.index, "role": "assistant", "content": turn.content}, None
            )
            s2.add(turn)
            await s2.flush()
            for pos, tid, ev in steps:
                s2.add(db.Step(turn_id=turn.id, position=pos, token_id=tid, evidence_hash=ev))
            conv2.next_turn += 1
            conv2.updated_at = time.time()
            await s2.commit()
            done = {
                "turn": turn.index,
                "engine": str(body.engine),
                "envelope_id": turn.envelope_id,
                "seed": turn.seed,
                "steps": len(steps),
                "evidence_kind": kind,
                "finish_reason": finish,
                "reproducible": body.engine == EngineId.REPRODUCIBLE,
            }
        yield _sse("done", done)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _chain(conv: db.Conversation, turn: db.Turn) -> None:
    prev = conv.turns[-1].hash if conv.turns else None
    turn.prev_hash = prev
    turn.hash = hash_event({"index": turn.index, "role": turn.role, "content": turn.content}, prev)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
