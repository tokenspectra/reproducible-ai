---
status: accepted
---
# `verify` is a separate entry point that reads only the trace and the weights

The verification plan's verifier is an adversarial third party with its own hardware and no access to the operator's runtime. We mirror that: `record` (operator side) and `verify` (verifier side) share the engine library, but `verify` is forbidden by construction from reading anything other than the trace file, the weights path and the environment it runs in. Any time `verify` would need a piece of operator state, that is a format bug and the test suite surfaces it.

## Considered options

One CLI that records and replays (lets operator state leak into replay unnoticed).
