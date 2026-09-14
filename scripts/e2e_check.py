#!/usr/bin/env python3
"""End-to-end check against the live stack.

Creates a conversation, sends a message, forks it, sends the same message to
both, and asserts the two answers are identical. Then issues a resume token,
imports it as a new conversation and continues from there. This is the demo,
run headlessly.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time

import httpx


def identity(secret: str, email: str, name: str) -> str:
    expires = int(time.time()) + 300
    msg = f"{email}\n{name}\n{expires}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"{email}|{name}|{expires}|{sig}"


def stream_message(c: httpx.Client, base: str, conv_id: str, content: str, engine: str) -> dict:
    text, done = [], {}
    with c.stream(
        "POST",
        f"{base}/v1/conversations/{conv_id}/messages",
        json={"content": content, "engine": engine, "max_tokens": 120},
        timeout=300.0,
    ) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if payload.get("text"):
                    text.append(payload["text"])
                if event == "done":
                    done = payload
                if event == "error":
                    raise RuntimeError(payload.get("message", "stream error"))
    return {"text": "".join(text), **done}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="https://api.overversion.com")
    p.add_argument("--secret", required=True)
    p.add_argument("--email", default="e2e@overversion.test")
    p.add_argument("--engine", default="reproducible")
    args = p.parse_args()

    headers = {"X-Overversion-Identity": identity(args.secret, args.email, "e2e")}
    with httpx.Client(headers=headers, timeout=60.0) as c:
        status = c.get(f"{args.base}/v1/status").json()
        print("default engine:", status["default_engine"])
        for e in status["engines"]:
            print(f"  {e['id']:<14} up={e['up']}")
        if not any(e["id"] == args.engine and e["up"] for e in status["engines"]):
            print(f"engine {args.engine} is down; nothing to test")
            return 2

        conv = c.post(f"{args.base}/v1/conversations", json={}).json()
        print(f"\nconversation {conv['id']} seed_source={conv['seed_source']}")

        first = stream_message(c, args.base, conv["id"], "Name three prime numbers.", args.engine)
        print(f"  turn 1: {first['steps']} steps, seed {first.get('seed')}")

        fork = c.post(f"{args.base}/v1/conversations/{conv['id']}/fork").json()
        print(f"forked to {fork['id']}")

        probe = "Now explain in two sentences why they are prime."
        a = stream_message(c, args.base, conv["id"], probe, args.engine)
        b = stream_message(c, args.base, fork["id"], probe, args.engine)

        same = a["text"] == b["text"]
        print(f"\nfork answers identical: {same}")
        print(f"  seeds:  {a.get('seed')} vs {b.get('seed')}")
        if not same:
            j = next(
                (i for i, (x, y) in enumerate(zip(a["text"], b["text"], strict=False)) if x != y),
                0,
            )
            print(f"  first difference at character {j}")
            print(f"    A ...{a['text'][max(0, j - 40) : j + 40]!r}")
            print(f"    B ...{b['text'][max(0, j - 40) : j + 40]!r}")
            return 1

        tok = c.get(f"{args.base}/v1/conversations/{conv['id']}/token").json()
        print(f"\ntoken issued: {tok['token'][:52]}… ({len(tok['token'])} chars)")

        other = {"X-Overversion-Identity": identity(args.secret, "guest@overversion.test", "guest")}
        with httpx.Client(headers=other, timeout=60.0) as g:
            imported = g.post(
                f"{args.base}/v1/conversations", json={"resume_token": tok["token"]}
            ).json()
            print(
                f"another account resumed it: {imported['id']} "
                f"turns={len(imported['turns'])} seed_source={imported['seed_source']}"
            )
            cont = stream_message(g, args.base, imported["id"], probe, args.engine)
        print(f"resumed answer matches the original: {cont['text'] == a['text']}")
        return 0 if cont["text"] == a["text"] else 1


if __name__ == "__main__":
    sys.exit(main())
