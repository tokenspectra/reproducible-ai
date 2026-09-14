#!/usr/bin/env python3
"""Chaos batcher: does a request's output depend on who it is batched with?

Sends one probe request many times while flooding the server with neighbour
requests of random lengths, so the probe lands in a different batch each time.
Deterministic mode should return byte-identical text every time; the baseline
instance should not.

Usage:
  python scripts/chaos_check.py --url https://infer.overversion.com --key "$SGLANG_API_KEY"
  python scripts/chaos_check.py --url https://infer.overversion.com/baseline --key ... --no-seed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import sys

import httpx

PROBE = "Explain in exactly three sentences why floating point addition is not associative."
NEIGHBOURS = [
    "Write a haiku about {}.",
    "List {} facts about the ocean.",
    "Summarise the history of {} in a paragraph.",
    "What is {}? Answer briefly.",
    "Translate 'good morning' into {}.",
]
TOPICS = ["copper", "Lisbon", "sailing", "entropy", "olives", "violins", "tides", "basalt"]


async def chat(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    model: str,
    prompt: str,
    *,
    seed: int | None,
    max_tokens: int,
    think: bool,
) -> str:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if seed is not None:
        body["seed"] = seed
    r = await client.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def one_round(
    client: httpx.AsyncClient, args, rng: random.Random, round_index: int
) -> str:
    """Fire the probe alongside a random crowd; return the probe's output."""
    crowd = [
        chat(
            client,
            args.url,
            args.key,
            args.model,
            rng.choice(NEIGHBOURS).format(rng.choice(TOPICS)),
            seed=rng.randrange(2**31) if not args.no_seed else None,
            max_tokens=rng.choice([16, 64, 256]),
            think=args.think,
        )
        for _ in range(rng.randrange(args.min_crowd, args.max_crowd + 1))
    ]
    probe = chat(
        client,
        args.url,
        args.key,
        args.model,
        PROBE,
        seed=None if args.no_seed else args.seed,
        max_tokens=args.max_tokens,
        think=args.think,
    )
    results = await asyncio.gather(probe, *crowd, return_exceptions=True)
    out = results[0]
    if isinstance(out, BaseException):
        raise out
    print(f"  round {round_index + 1}: crowd={len(crowd)} sha={sha(out)}", flush=True)
    return out


def sha(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="base url, e.g. https://infer.overversion.com")
    p.add_argument("--key", required=True)
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--min-crowd", type=int, default=0)
    p.add_argument("--max-crowd", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--no-seed", action="store_true", help="omit the seed (baseline behaviour)")
    p.add_argument("--think", action="store_true", help="leave Qwen3 thinking mode on")
    args = p.parse_args()

    rng = random.Random(0)
    print(f"probe x{args.rounds} against {args.url} (seed={'none' if args.no_seed else args.seed})")
    async with httpx.AsyncClient() as client:
        outputs = [await one_round(client, args, rng, i) for i in range(args.rounds)]

    unique = {sha(o) for o in outputs}
    print(f"\nunique outputs: {len(unique)} of {len(outputs)}")
    if len(unique) == 1:
        print("PASS: identical under every batch composition")
        return 0
    print("DIFFER: batch composition changed the output")
    first = outputs[0]
    for i, o in enumerate(outputs[1:], start=1):
        if o != first:
            j = next((k for k, (a, b) in enumerate(zip(first, o, strict=False)) if a != b), min(len(first), len(o)))
            print(f"  round 1 vs {i + 1}: first difference at character {j}")
            print(f"    round 1  ...{first[max(0, j - 40):j + 40]!r}")
            print(f"    round {i + 1}  ...{o[max(0, j - 40):j + 40]!r}")
            break
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
