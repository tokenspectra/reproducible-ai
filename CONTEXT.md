# reproducible-ai

Bitwise-deterministic LLM inference with self-contained, replayable traces that
an independent verifier can re-execute. This glossary fixes the words used in
the design, the code and the docs.

## Language

### Determinism

**Determinism claim**:
The guarantee that a request yields bitwise-identical logits at every step on a
fixed envelope, independent of batch composition, restarts, chunking, or the
prefill-vs-decode path.
_Avoid_: reproducibility (too broad), stability

**Batch invariance**:
The property of an op or engine that a request's outputs do not depend on which
other requests share the batch or on the batch size.
_Avoid_: batch determinism, shape invariance

**Envelope**:
The set of hardware and software facts under which the determinism claim holds.
Split into hard fields (mismatch refuses a verify) and soft fields (recorded,
compared, never blocking).
_Avoid_: environment, config, fingerprint

**Manifest**:
The part of the header that tells a verifier how to rebuild the stack: engine,
git SHA, lockfile hash, and from the cloud phase a container image digest. The
"instructions" of a packet.
_Avoid_: lockfile, image spec

**Kernel registry**:
The set of custom kernels with stable ids, whose compiled device binaries are
hashed together into the `kernel_registry_hash` hard envelope field.
_Avoid_: kernel cache, op table

**Divergence**:
The first step at which a replay's evidence differs from the trace.
_Avoid_: mismatch (reserved for envelope fields), drift

### Trace

**Trace**:
The session-level artifact: a header followed by an append-only, hash-chained
stream of events, one JSON object per line.
_Avoid_: log, transcript, recording, session file

**Turn**:
One message in a trace. An assistant turn carries its own seeds, sampling
params and steps.
_Avoid_: message, round, exchange

**Step**:
One sampled token within an assistant turn, identified by its absolute position
in the sequence and carrying the token id and the logits hash.
_Avoid_: timestep, iteration, token record

**Position**:
The absolute index of a token in the full token sequence of the session. The
coordinate used by the RNG contract in place of any "timestep".
_Avoid_: step index, offset, timestep

**Packet**:
A self-contained, independently replayable unit cut from a trace: header,
prefix tokens, a step range and their evidence. The verification plan's unit of
recomputation.
_Avoid_: chunk, segment, batch

**Evidence**:
The per-step data a verifier compares: token id, logits hash, and optionally
layer hashes or a sidecar.
_Avoid_: output, result, proof

**Logits hash**:
BLAKE3-128 of the fp32, pre-temperature logits in canonical byte order; the
`logits_hash_kind` field says when an engine could only expose a derived
quantity.
_Avoid_: checksum, fingerprint, signature

**Sidecar**:
A safetensors file holding debug payloads (full logits, hidden states) that a
step record references instead of embedding.
_Avoid_: dump, attachment

### Seeds and randomness

**Root seed**:
The 256-bit per-session seed in the header from which all other seeds derive.
_Avoid_: master seed, global seed, session seed

**Seed source**:
Whether the root seed came from OS randomness or was supplied externally (for
example by an auditor).
_Avoid_: seed origin, provenance

**Turn seed**:
The per-turn seed derived from the root seed and the turn index, from which the
sampling seed and the SR seed derive. Each is recorded and independently
overridable.
_Avoid_: request seed, generation seed

**Element identity**:
The batch-invariant coordinates that key every random draw: for stochastic
rounding (tensor id, position, feature index); for sampling (position, draw
index). Never a location in a batched tensor.
_Avoid_: element index, thread id, offset

**Tensor id**:
A stable identifier of a tensor in the model graph (layer index, op name,
output slot), used in element identity.
_Avoid_: tensor name, buffer id, pointer

**Stochastic rounding**:
Rounding a higher-precision value to a lower-precision format with a
probability proportional to the remainder, using a draw keyed by element
identity so the rounding mask is batch-invariant.
_Avoid_: SR (in prose), random rounding, dithering

### Execution

**Record**:
The operator-side entry point that runs requests through an engine and writes
traces.
_Avoid_: serve, generate, capture

**Verify**:
The verifier-side entry point that takes a trace and weights, checks the
envelope and hash chain, replays, and reports pass or divergence. It reads
nothing else.
_Avoid_: replay (that is one thing verify does), check, validate

**Fast resume**:
Rebuilding a trace's state by prefilling the recorded prefix in one shot.
_Avoid_: prefill resume, warm start

**Exact replay**:
Rebuilding a trace's state by re-executing every recorded decode step one token
at a time. The oracle against which fast resume is tested.
_Avoid_: step replay, slow path

**Reference oracle**:
The minimal plain-torch engine in this repo that exists to make the trace
format and verifier testable end to end; not a serving engine.
_Avoid_: reference engine, toy engine, our engine

**Adapter**:
The glue that lets `record` and `verify` drive an external engine (SGLang
first, vLLM later).
_Avoid_: backend, plugin, integration

**Chaos batcher**:
The test harness that runs a request alongside randomly generated neighbors of
random lengths, batch sizes, arrival orders and preemptions.
_Avoid_: fuzzer, load test, stress test

**Dtype config**:
The recorded precision contract of a run: storage dtypes, accumulation dtype,
KV-cache dtype, output cast mode and quantization tier.
_Avoid_: precision, quant config, dtype
