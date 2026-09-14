import json

import httpx
import pytest
from fastapi.testclient import TestClient

from overversion_api.app import create_app
from overversion_api.auth import make_header
from overversion_api.config import Settings
from tests.test_status import SERVER_INFO_DET

SECRET = "shared-secret"
SIGNING_SEED = "11" * 32


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


SENT_BODIES: list[dict] = []


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/server_info"):
        return httpx.Response(200, json=SERVER_INFO_DET)
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json={"data": {}})
    if request.url.path.endswith("/chat/completions"):
        body = json.loads(request.content)
        SENT_BODIES.append(body)
        if not body.get("messages"):
            return httpx.Response(400, json={"message": "Messages cannot be empty."})
        seed = body.get("seed", 0)
        return httpx.Response(
            200,
            content=sse(
                {
                    "choices": [
                        {
                            "delta": {"content": f"answer-for-seed-{seed}"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {
                                        "token_id": 5,
                                        "top_logprobs": [{"token_id": 5, "logprob": -0.1}],
                                    }
                                ]
                            },
                        }
                    ]
                }
            ),
        )
    return httpx.Response(404)


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        infer_base_url="https://infer.test",
        infer_api_key="k",
        openrouter_api_key="or",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/t.db",
        frontend_shared_secret=SECRET,
        token_signing_seed_hex=SIGNING_SEED,
        status_interval_s=3600.0,
    )
    import httpx as _httpx

    real = _httpx.AsyncClient

    def patched(*a, **kw):
        kw["transport"] = _httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(_httpx, "AsyncClient", patched)
    app = create_app(settings)
    with TestClient(app) as c:
        c.headers.update({"X-Overversion-Identity": make_header(SECRET, "a@b.c", "A")})
        yield c


def test_requires_identity(client):
    r = client.get("/v1/conversations", headers={"X-Overversion-Identity": ""})
    assert r.status_code == 401


def test_bad_signature_rejected(client):
    bad = make_header("wrong-secret", "a@b.c", "A")
    r = client.get("/v1/conversations", headers={"X-Overversion-Identity": bad})
    assert r.status_code == 401


def test_create_send_and_fork(client):
    conv = client.post("/v1/conversations", json={}).json()
    assert conv["next_turn"] == 0 and conv["seed_source"] == "os_random"

    with client.stream(
        "POST",
        f"/v1/conversations/{conv['id']}/messages",
        json={"content": "hello", "engine": "reproducible"},
    ) as r:
        assert r.status_code == 200
        events = [line for line in r.iter_lines() if line.startswith(("event:", "data:"))]
    assert any("answer-for-seed-" in e for e in events)
    assert any('"reproducible": true' in e.replace("True", "true") for e in events)

    full = client.get(f"/v1/conversations/{conv['id']}").json()
    assert [t["role"] for t in full["turns"]] == ["user", "assistant"]
    assert full["turns"][1]["engine"] == "reproducible"
    assert full["turns"][1]["seed"] is not None
    assert full["turns"][1]["steps"] == 1

    fork = client.post(f"/v1/conversations/{conv['id']}/fork").json()
    assert fork["forked_from"] == conv["id"]
    assert len(fork["turns"]) == 2
    assert fork["next_turn"] == full["next_turn"]


def test_fork_replays_same_seed(client):
    """The point of a fork: the same next message gets the same seed."""
    conv = client.post("/v1/conversations", json={}).json()
    client.post(
        f"/v1/conversations/{conv['id']}/messages", json={"content": "q", "engine": "reproducible"}
    )
    fork = client.post(f"/v1/conversations/{conv['id']}/fork").json()
    for cid in (conv["id"], fork["id"]):
        client.post(
            f"/v1/conversations/{cid}/messages", json={"content": "same", "engine": "reproducible"}
        )
    a = client.get(f"/v1/conversations/{conv['id']}").json()["turns"][-1]
    b = client.get(f"/v1/conversations/{fork['id']}").json()["turns"][-1]
    assert a["seed"] == b["seed"]
    assert a["content"] == b["content"]


def test_token_roundtrip_and_import(client):
    conv = client.post("/v1/conversations", json={}).json()
    assert client.get(f"/v1/conversations/{conv['id']}/token").status_code == 409
    client.post(
        f"/v1/conversations/{conv['id']}/messages", json={"content": "q", "engine": "reproducible"}
    )
    tok = client.get(f"/v1/conversations/{conv['id']}/token").json()
    assert tok["token"].startswith("rai1.")
    assert client.post("/v1/tokens/inspect", json={"token": tok["token"]}).json()["valid"]

    # default: seed only, nothing copied
    seeded = client.post("/v1/conversations", json={"resume_token": tok["token"]}).json()
    assert seeded["imported_from_token"] is True
    assert seeded["seed_source"] == "external"
    assert seeded["turns"] == []
    assert seeded["next_turn"] == 0

    # opting in to the full state copies the history and continues from it
    imported = client.post(
        "/v1/conversations", json={"resume_token": tok["token"], "token_mode": "state"}
    ).json()
    assert len(imported["turns"]) == 2
    assert imported["forked_from"] == conv["id"]
    assert imported["next_turn"] == 2


def test_tampered_token_rejected(client):
    conv = client.post("/v1/conversations", json={}).json()
    client.post(
        f"/v1/conversations/{conv['id']}/messages", json={"content": "q", "engine": "reproducible"}
    )
    tok = client.get(f"/v1/conversations/{conv['id']}/token").json()["token"]
    head, payload, sig = tok.split(".")
    r = client.post("/v1/conversations", json={"resume_token": f"{head}.{payload}x.{sig}"})
    assert r.status_code == 400


def test_explicit_seed_recorded_as_external(client):
    conv = client.post("/v1/conversations", json={"seed_hex": "ab" * 32}).json()
    assert conv["seed_source"] == "external"


def test_send_to_down_engine_is_refused(client, monkeypatch):
    conv = client.post("/v1/conversations", json={}).json()
    import time as _time

    from overversion_api.engines import EngineId

    tracker = client.app.state.status
    tracker._statuses[EngineId.REPRODUCIBLE].up = False
    tracker._statuses[EngineId.REPRODUCIBLE].detail = "box is off"
    tracker._checked_at = _time.time()  # keep it fresh so current() serves this
    r = client.post(
        f"/v1/conversations/{conv['id']}/messages",
        json={"content": "x", "engine": "reproducible"},
    )
    assert r.status_code == 409 and "box is off" in r.text


def test_cannot_read_another_users_conversation(client):
    conv = client.post("/v1/conversations", json={}).json()
    other = make_header(SECRET, "other@b.c", "O")
    r = client.get(
        f"/v1/conversations/{conv['id']}", headers={"X-Overversion-Identity": other}
    )
    assert r.status_code == 403


def test_history_sent_upstream_includes_the_new_message(client):
    """The engine must receive the conversation so far plus the message just sent.

    A reload of the conversation here returns the identity-mapped object whose
    turns predate the insert, which sends an empty history and makes the
    upstream reject the request.
    """
    SENT_BODIES.clear()
    conv = client.post("/v1/conversations", json={}).json()
    client.post(
        f"/v1/conversations/{conv['id']}/messages",
        json={"content": "first question", "engine": "reproducible"},
    )
    assert [m["content"] for m in SENT_BODIES[-1]["messages"]] == ["first question"]

    client.post(
        f"/v1/conversations/{conv['id']}/messages",
        json={"content": "second question", "engine": "reproducible"},
    )
    roles = [m["role"] for m in SENT_BODIES[-1]["messages"]]
    contents = [m["content"] for m in SENT_BODIES[-1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert contents[0] == "first question"
    assert contents[-1] == "second question"


def test_conversations_get_incremental_numbers(client):
    a = client.post("/v1/conversations", json={}).json()
    b = client.post("/v1/conversations", json={}).json()
    fork = client.post(f"/v1/conversations/{a['id']}/fork").json()
    assert [a["seq"], b["seq"], fork["seq"]] == [1, 2, 3]

    # numbers are per user, and deleting does not renumber or reuse
    client.delete(f"/v1/conversations/{b['id']}")
    c = client.post("/v1/conversations", json={}).json()
    assert c["seq"] == 4

    other = make_header(SECRET, "second@b.c", "S")
    mine = client.post(
        "/v1/conversations", json={}, headers={"X-Overversion-Identity": other}
    ).json()
    assert mine["seq"] == 1


def test_token_as_seed_reproduces_the_same_first_answer(client):
    """The headline demo: two separate empty chats, same first message, same answer."""
    first = client.post("/v1/conversations", json={}).json()
    client.post(
        f"/v1/conversations/{first['id']}/messages",
        json={"content": "hola, poema por favor", "engine": "reproducible"},
    )
    tok = client.get(f"/v1/conversations/{first['id']}/token").json()["token"]

    second = client.post("/v1/conversations", json={"resume_token": tok}).json()
    assert second["turns"] == []
    client.post(
        f"/v1/conversations/{second['id']}/messages",
        json={"content": "hola, poema por favor", "engine": "reproducible"},
    )

    a = client.get(f"/v1/conversations/{first['id']}").json()["turns"][1]
    b = client.get(f"/v1/conversations/{second['id']}").json()["turns"][1]
    assert a["seed"] == b["seed"], "same root seed and same turn index must give the same subkey"
    assert a["content"] == b["content"]
