---
status: accepted
---
# The trace is a session of hash-chained JSONL events, not a per-request record

Resume-then-continue with a new message, and the verification plan's agentic multi-turn scope, both make the natural unit a conversation. A trace is therefore a header line plus append-only events (turn start, step, turn end), one JSON object per line, each carrying the hash of the previous event. The chain is for integrity and tamper evidence, not proof: the verifier is adversarial and only trusts re-execution. Per-step records carry absolute position, token id and the logits hash, so a step range is a self-contained packet; a separate `packet` command is deferred. Large debug payloads go to a safetensors sidecar.

## Considered options

One trace per request with parent references (awkward for multi-turn resume). Single JSON rewritten on append (not append-only). Binary container from day one (premature; can wrap JSONL later without a schema change).
