# AI 2040 "Verification Plan" — requirements relevant to deterministic, replayable inference

- **Source:** https://ai-2040.com/supplements/verification-plan (supplement to "AI 2040: Plan A", author Romeo Dean)
- **Fetched:** 2026-09-09 (UTC), full page HTML retrieved and converted to text; all quotes below are verbatim from the page.
- **Why this matters to us:** the page is the design brief that motivated building bitwise-deterministic inference with replayable traces. It describes a scheme in which an *adversarial* verifier recomputes randomly sampled "packets" of AI work on its own hardware and compares the result to what the operator produced. Everything below is filtered for what that implies about an inference trace format and a determinism envelope.

---

## 1. Overall purpose: who verifies what, and why replay matters

**The setting is a US–China treaty to slow frontier AI, enforced "without relying on trust."**

> "Plan A is predicated on the ability to enforce an international deal to slow down AI progress without relying on trust. We are confident that verification measures can make this possible, by providing sufficient assurance to both the US and China that defection by the other would either be caught or small enough to be inconsequential."

The verification problem is split in two; only the first concerns us:

> "Problem 1. Verifying that known compute is compliant. ... Problem 2. Verifying that unknown compute is inconsequential."

**What is being verified:** that datacenters are running only approved *inference* (and later, only approved R&D workloads) and not training/experiments. The recommended mechanism is "inference-only verification" via a hardware retrofit:

> "Our current best guess is that the US and China should aim for an inference-only solution that uses network taps and a recomputation server."

> "Whatever the best inference-only verification solution that is ready should be used. We currently recommend aiming for a network-tap and recomputation based approach, but more R&D is needed to figure out if another approach will be better."

**Why determinism/replay is load-bearing.** The verifier's algorithm is *partial recomputation of randomly sampled packets*. That only works if a packet, re-executed elsewhere, yields the same answer:

> "Force all AI workloads (inference, agentic tool use, etc.) to be organized in discrete and reproducible packets, which are transparent to the recomputation server (because they pass through the network taps). Reproducibility is required to enable correctness proofs on the packets. By default, inference workloads are not necessarily reproducible. Achieving reproducibility may involve some cost overhead in order to maintain reproducibility but we think it is a solvable problem (see here for example of work in this direction)."

The "see here" link points to Thinking Machines' *Defeating Nondeterminism in LLM Inference* (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and to arXiv 2511.20621 (https://arxiv.org/pdf/2511.20621).

> "Perform partial recomputation (random samples of the packets) on the recomputation server to check the correctness of the packets. The integrity of the recomputation server is an extremely important aspect of this step, but assuming its computations and outputs can be trusted, then partial recomputation through random sampling should scale extremely favourably to verify the correctness of all the outputs with extremely high probability while only using a small fraction of the original compute."

> "By default, these computational steps execute with randomness, but if they can be forced to be reproducible (e.g., seeded randomness is fine), then we can have a predictable relationship between the granularity of the evidence collection and the assurance level of the verification algorithm."

> "Our guess is that in order to reach extremely high levels of assurance ... a reproducibility mandate, combined with complete (and increasingly granular) on-path evidence collection, paired with a partial recomputation based verification algorithm, will be a viable approach."

**Why it needs to be cheap and ready in advance.** The whole economic case rests on keeping inference online during a deal; the difference between having and not having a ready inference-only verification solution is estimated in the trillions of dollars:

> "If there were a minimum-viable inference-only retrofitting solution prepared, the equivalent period would be more like 2-5 months [vs 6-12 months of ~90% of frontier services offline]. The difference between these in terms of economic impact might be in the trillions of dollars by 2029."

**The asymmetry that makes sampling work** (footnote 5):

> "the verifier only needs to catch one violation, while a prover needs to successfully hide all of their covert computations. This is the reason behind e.g., the favourable scaling of partial recomputation due to random sampling."

---

## 2. Concrete technical requirements

### 2.1 The unit of verification: the "packet"

The appendix gives the closest thing to a formal definition of the trace unit:

> "A packet is a reproducible unit of computation that the verifier can recompute: its inputs, instructions, and outputs are all collected at the evidence-collection point (e.g., a network tap). What the computation actually is (one forward pass, one training batch) doesn't affect the math below."

So a packet must carry at least:

| Field | Source on the page |
|---|---|
| **Inputs** | "its inputs, instructions, and outputs are all collected" |
| **Instructions** (what computation to run: model, code, algorithm) | same; plus Declaration step: "AI companies declare the workload they want to run with full transparency on the code, algorithms, etc." |
| **Outputs** | same; for inference, "verified inference tokens" |
| **Random seeds** | "seeded randomness is fine"; and in the R&D regime the auditor *supplies* them: "provide necessary random seeds (removes a potential attack vector from AI companies gaming fake randomness)" |
| **Model identity (whitelisted weights)** | For secure gateways: "In a boot phase, these gateways can additionally check that exclusively whitelisted model weights are being uploaded to the inference unit." |

For training packets the page is explicit about contents:

> "Network taps between server racks log every gradient step (its input batch, the gradient update, and the resulting checkpoint update), so each gradient step is one packet."

**Not mentioned anywhere on the page:** logits, model hashes, software/stack versions, hardware identifiers, driver/kernel versions. The page speaks at the level of "inputs, instructions, outputs" and leaves the concrete encoding open. Anything needed for re-execution that does not naturally cross the network (seed, sampling parameters, model version, stack version) must therefore either (a) be pushed onto the wire so the tap captures it, or (b) be fixed by the declaration/approval step. The page's design leans on (b): the workload is declared and approved up front, and the auditor issues the seeds.

### 2.2 What must be reproducible, and how exactly

- The page requires reproducibility of **outputs** ("verify the correctness of all the outputs"). For inference the outputs are **tokens**: the completeness goal is "that the only outputs from the cluster are verified inference tokens." Logits are never mentioned; bitwise equality is never mentioned as such, but the appendix's "key simplifying assumption ... is that the workload packets are perfectly reproducible," and the verification algorithm has no tolerance concept, so the working assumption is exact-match recomputation.
- The **stochastic parts must be seeded** ("seeded randomness is fine"), and the seeds should be **verifier-chosen** (see the auditor-provides-seeds quote above). That is a strictly stronger requirement than "record the seed": the trace format should accept an externally supplied seed and the sampler must be a deterministic function of (logits, seed).
- **Cross-machine reproducibility is mandatory.** The recomputation is done on a different machine, owned by the other party ("unilaterally trusted recomputation servers"). So reproducibility must hold across at least the operator's inference hardware and the verifier's recomputation server. The page acknowledges hardware heterogeneity as the thing that makes reproducibility feasible for AI and infeasible for lab science: "two labs in the real world are not as homogenous as AI chips." It also anticipates needing to *force* homogeneity: "Some degree of standardization in hardware may also be helpful for reproducibility," and "Huge progress in standardizing (for reproducibility) and improving security of software stacks for inference and R&D workloads more generally" is listed as a 2033 improvement. Phase 3 hardware goals include "extremely high fidelity reproducibility."
- **Agentic tool use is in scope:** "Force all AI workloads (inference, agentic tool use, etc.) to be organized in discrete and reproducible packets." A trace format must therefore handle multi-turn / tool-call loops, not only single forward passes.
- Overhead is accepted but should be small: "Achieving reproducibility may involve some cost overhead"; the page's economic-cost metric says the dominant cost is how much inference is allowed to keep running, not the verification machinery itself.

### 2.3 Who runs the replay

An **adversarial third party on its own hardware** — specifically the other treaty party, with equipment it built and trusts unilaterally:

> "Unilaterally trusted recomputation servers (that the network taps redirect traffic to and are responsible for actually performing the verification) are more complicated and thus harder to trust by default, but there are large investments made here. The R&D programs are purposely transparent and open sourced to build trust. China also sees the incentive to develop and build their own symmetrical solution in a similarly transparent way."

> "Cankaya proposes using passive optical fiber splitters (no digital logic, just fused glass) so both parties can independently observe network traffic without needing to trust the other's hardware, combined with unilaterally trusted devices to independently implement their own verification."

The recomputation servers sit inside the prover's datacenter ("The prospect of installing devices in each other's datacenters raises initial alarm bells"). In the R&D regime, the roles are: companies declare, "Auditors run checks ... and if it passes they send back approval and provide necessary random seeds," "Auditors collect evidence from the workload (e.g., complete logging flow through network taps)," and "Auditors sample randomly and do partial recomputation and other checks to catch violations of workload correctness."

Hardware requirements on the recomputation server (2033 list): "Generally improved hardware security properties with more robust trusted execution, secure boot, encryption, reproducibility, etc., which are especially important for recomputation servers."

### 2.4 Evidence collection: network taps on the frontend network

> "Add in simple, passive network taps on the frontend network (also known as north-south networking) that redirect copies of all input/output traffic to a recomputation server. ... We currently think they should probably be completely passive and split optical signals into two because this is probably easiest to trust. Separately there should be a path from the storage bank back to the inference units."

Implications for the trace: **the trace is whatever crosses the wire.** The tap copies "all input/output traffic"; the recomputation server must be able to reassemble that traffic into packets ("organize the logs of reproduction packets"). The trace format therefore needs to be self-delimiting and reconstructable from a passive bytestream, without cooperation from the operator at reconstruction time. The "path from the storage bank back to the inference units" also implies the trace/storage must support feeding stored inputs back for re-execution.

A later upgrade, "secure network gateways," would do sampling at the edge and check weights at boot:

> "There are data path directions (input and output) with network splitters on each path that log data flow to be organized into packets that are randomly sampled and then sent to a secure recomputing server. In a boot phase, these gateways can additionally check that exclusively whitelisted model weights are being uploaded to the inference unit. Some of the hardware security properties you might want for these gateways are precedented, for example, see iPhone secure enclaves and PUFs."

### 2.5 Granularity: packet size, and how it shrinks over time

The assurance math depends on packet size `s` (fraction of pool) and recomputation budget `C`:

> "Since N_verified = C / s, the curve can be shifted higher either by (1) increasing the recomputation budget C or (2) shrinking the packet size s."

> "The product C · N_fake (recomputation budget times number of fake packets) is the single quantity that determines 99% catch probability: it just needs to exceed about 4.6."

> "random sampling buys you a lot once you're catching attacks that span at least dozens of packets. To catch attacks below that, shrink packets (decrease p) so the same fake compute-work spans more packets."

Concrete operating points on the page:

| Year / stage | Evidence-collection point | Packet size |
|---|---|---|
| Feb 2029 initial retrofit | frontend taps around "inference units of 4 server racks (or 4K H100e)" | inference-unit level |
| Aug–Sep 2029 | "scaled to higher granularity (smaller inference units)" | smaller |
| R&D clusters, 2029+ | "smaller and smaller packet sizes over time by moving the network taps to lower levels of the backend network"; "could be happening at the level of individual servers, or GPUs or at even lower levels later on" | rack → server → GPU |
| 2034 | "one packet collected per hour from each frontier GPU (~100 H100e each, so ~100 H100e-hour packets) and a 1% recomputation budget" | 100 H100e-hours |
| Appendix table | | 100, 10, 1 H100e-hour with C = 1% or 0.1% |
| Training worked example | per gradient step; "sub-batch logging, tracking intermediate computations within a batch (e.g., individual forward/backward passes), which might require evidence-collection devices inside server racks or even inside GPUs themselves" | one gradient step → sub-batch |

Interactive figure captions on the page show the parameter sweep: "Pool: ~33B H100e (2034 buildout) / Recomputation budget: 1% / Year: 2030 2032 2034 2040 / Packet size: 1 10 100 1K 10K H100e-hours."

Design consequence: the trace format should not hard-code one granularity. It should be possible to verify a request, a single forward pass, or (eventually) a sub-step of a forward pass, with the same "inputs, instructions, outputs, seed" structure.

### 2.6 Recomputation budget and detection bound (latency/storage proxies)

The page never states a latency requirement (verification is offline/after-the-fact) and never states a storage budget. The constraints that do appear:

- **Recomputation budget:** 1% is the working figure (0.1% and 1% in tables; the training example shows 0.07%–70% depending on batch size).
- **Detection bound:** "the verifier catches with 99% confidence any rogue internal deployment that accumulates more than about 46K H100e-hours of unapproved compute work inside the verified pool, regardless of pace or distribution" (p = 100 H100e-hours, C = 1%). "so a sustained 10K H100e deployment is detected within ~4.6 hours, a 1K within ~46 hours."
- **Logging bandwidth:** a 2033 goal is "Increases to the logging / evidence collection granularity of the verification regime ... with logging technology being compatible with extremely high bandwidth data paths on backend networking to reduce overhead."
- **Pool-level sampling:** footnote 20: "none of these quantities depend on how the work is divided into individual workloads. What matters is the pool the verifier samples from. If there are 100 different training runs going on simultaneously and the verifier samples uniformly at random across all their packets, then 'the workload' in the math below is the union of all 100." So packets must be independently sampleable and re-executable out of context (a random packet from the middle of a session must be replayable without replaying everything before it — i.e., the packet must carry or reference its full input state).

Storage/retention interacts with the privacy tiers (see §5): "Total ZDR" promises 100% deletion, which is in tension with taps copying all traffic to a verifier; the page does not resolve this.

### 2.7 Cryptography, attestation, TEEs

The page is notably **light on cryptography** and deliberately prefers dumb physical mechanisms:

> "Ultimately the US and China end up opting for network tap solutions with extremely simple optical splitting designs that have minimal room for any vulnerabilities, rather than some of the more ambitious designs that we propose in the box below, or different cryptographic based-approaches, which are more uncertain, but if they work (at sufficiently low cost) they might be even better and easier to trust."

- **ZKPs:** listed as an alternative, "currently somewhat speculative due to the lack of computationally efficient enough algorithms." ~$100M R&D suggested for "software-only and cryptography-centric (e.g., ZKPs) approaches."
- **TEEs / secure boot / encryption:** appear only as future hardware improvements "especially important for recomputation servers" (2033), and for gateways ("iPhone secure enclaves and PUFs").
- **Hardware-derived keys** for weights: "build computing clusters that keep model weights encrypted inside secure parts of the AI clusters using hardware-derived keys that are unobservable from the outside."
- **Offline licensing** as a crypto tool for mutually assured compute destruction.
- **Not mentioned at all:** hashing, digital signatures, commitments, Merkle trees, remote attestation protocols. The page's proof-of-correctness is *re-execution and comparison*, not a signed claim. A hash or signature from the operator would be worthless under this threat model since the operator is the adversary; the verifier must re-derive the output itself. (Hashes could still be useful *inside* the verifier's pipeline for comparing recomputed vs. observed outputs, but the page does not go there.)

### 2.8 Sampling, batching, multi-GPU, quantized formats

- **Sampling (token sampling):** only "seeded randomness is fine" and auditor-provided seeds. No mention of temperature/top-p.
- **Batching:** not discussed for inference. For training, batch size is the packet-size lever: "Making batches 10× smaller cuts the required recomputation cost 10× for the same confidence." (The linked Thinking Machines post is about batch-invariant kernels, which is the relevant inference-side concern, but the page itself does not say "batch-invariant.")
- **Multi-GPU / interconnect:** the retrofit *removes* most scale-out networking: "Convert datacenters into groups of isolated inference-units by removing (a significant amount of) backend networking (also known as east-west networking). ... with some relatively small overhead (e.g., reorganizing user memory storage), large scale inference should be able to use drastically less scale-out interconnect." The initial inference unit is 4 racks / 4K H100e. So the determinism envelope must cover at least intra-unit (multi-GPU, multi-server) inference; the packet boundary is the unit's frontend network.
- **Quantized / number formats:** not discussed as a reproducibility issue. Number formats appear only as a deal-stability lever: "introduce new model architectures and number formats different from those for which pre-deal compute was optimized," and "hardware-software co-dependencies especially with architecture changes or number formats." Design note: the trace format should carry dtype/number-format info because the plan expects formats to change over the deal's lifetime.
- **Hardcoded weights:** "fixed set hardware designs (e.g., hardcoded weights might be a nice way to upgrade the inference-only verification)"; "hardcoded weights on some chips to verify they're inference-only."

---

## 3. Threat model

**Fully adversarial, nation-state-level.** The prover (AI company / host country) is assumed willing to cheat; the verifier trusts nothing the prover supplies except what physics guarantees (glass splitters) plus what the verifier recomputes on its own trusted hardware.

- Security target: "at least AI model weights and verification correctness requiring SL5 (robust to nation state actors) level security."
- The verifier does **not** trust the operator's hardware or software: taps are "completely passive and split optical signals into two because this is probably easiest to trust"; recomputation servers are "unilaterally trusted"; verification hardware itself needs supply-chain verification: "Many verification mechanisms will depend on hardware, e.g., network taps, recomputation servers, tamper-evident enclosures, etc., actually being what they claim to be. This creates a desire for mutual verification of the manufacturing and supply chain for verification hardware itself."
- **Explicit attacks the page defends against:**
  - *Packet substitution / hidden model or workload:* "Suppose an attacker substitutes N_fake packets, replacing the declared computation in each with their own unauthorized one." This is exactly the "operator ran a different model than declared" case.
  - *Fake randomness:* auditors provide seeds, which "removes a potential attack vector from AI companies gaming fake randomness." I.e. the operator must not be allowed to pick seeds after seeing inputs, or to claim any output was "just a random sample."
  - *Steganographic workloads:* "it might be possible for AI companies to do adversarial attacks, like trying to encode a non-compliant workload inside a workload that looks compliant on the surface." Mitigations: "Anti-steganography measures," "Auditor compilation into gpu instructions," whitelists/blacklists of code and data.
  - *Physical tampering:* "The integrity of the verification process requires there to be sufficient physical security measures to avoid tampering."
  - *Side channels / covert output:* "Verify completeness by bounding the maximum side channel communication, and using periodic memory wipes (probably through forced memorization)."
  - *Rogue internal deployments* inside verified clusters (A.4.2), and *backdoor insertion* in training (A.4.1: 100K off-spec tokens in a 100T-token run).
  - *Model weight theft* during release: "This process is extremely high stakes from a model weights theft perspective."
- **Consequence for trace design:** an operator-produced hash of logits (or a signature over outputs) does *not* satisfy this model. The verifier must be able to independently re-derive the outputs from inputs + instructions + seed on independent hardware and compare. The trace must be complete enough for that, and must be captured passively (the operator cannot be the one who decides what gets logged).
- The page does allow trust in *symmetry and openness* as a trust-building substitute: "The R&D programs are purposely transparent and open sourced to build trust."

---

## 4. Explicit non-goals and out-of-scope items

- **Does not verify that spare compute is idle:** "We do not verify that spare compute isn't used for unapproved workloads (because this seems very challenging) but try to ensure that the only persisting results in the form of outputs are verified inference outputs."
- **Assumes perfect reproducibility rather than establishing it:** "A key simplifying assumption we make in this analysis is that the workload packets are perfectly reproducible." Also assumes "Uniform size" packets and "Independent random sampling" (with-replacement) as simplifications; says non-uniform packets and without-replacement sampling work too.
- **Random sampling is useless against tiny attacks:** "To catch a single fake packet with 99% confidence, you'd need to recompute 99% of all packets, i.e. you've essentially given up on sampling and are doing full verification." Fix is smaller packets, not a different algorithm.
- **Workload approval is out of scope of the recomputation mechanism** and is done manually: "our overall baseline proposal for approval is for this to just be manually carried out by teams of auditors in the R&D clusters." Verification only checks that "Workloads that the companies declare match what they actually run."
- **Privacy is not a design goal of the primary mechanism:** "it would still be ideal for verification to not require low level access (e.g. tokens, model weights, algos)," but "We aren't overly worried about this because we are independently recommending increased transparency anyway." Privacy-preserving approaches are a 20% side-bet.
- **Cryptographic / ZKP approaches are not the primary path** (see §2.7).
- **Human methods are not load-bearing:** "We don't rely on these as load-bearing in any of our proposals."
- **Hardware research is expected to be largely non-reproducible** and thus out of scope for recomputation: "it seems far harder to get high reproducibility in e.g., a chemistry experiment ... Therefore, we probably need to not go with a reproducing based approach on everything."
- **Small clusters (<10K H100e) and consumer/edge compute** are handled by reporting, spot checks, caps and buy-backs, not by the retrofit.
- **Pre-deal chip tracking does not need location or usage:** "our recommendation for this early chip tracking doesn't prioritize knowing where this compute is, or what it is being used for."
- **Timing is not part of correctness.** The page never requires latency reproduction; "regardless of pace or distribution" appears in the detection bound.
- The author is explicit about uncertainty: "Because it is a scenario about the future, in a very young field with ongoing R&D, we are not confident in the specific details."

---

## 5. Other constraints and design informants

1. **Two-part framing: evidence collection vs. proving.** "divide it in two parts: (1) collecting evidence, and (2) proving something based on the evidence." A trace format is the interface between those halves and should be designed so either half can be swapped (taps → gateways; recomputation → ZKP later).

2. **Five evaluation metrics** the page uses for any verification solution: assurance curve (coverage × confidence), execution difficulty, preparation required, privacy level, economic cost, plus scope. Determinism work scores mostly on "preparation required" and "economic cost" (overhead).

3. **Assurance-curve math** to reuse when sizing packets:
   `P(undetected) = (1 − C)^N_fake ≈ e^(−C·N_fake)`; 99% catch needs `C·N_fake ≥ 4.6`; 99.9% needs ≥ 7; max undetected rogue work `W = 4.6·p / C`. Coverage `= 1 − F*_fake`, confidence `= 1 − e^(−N_verified·F*_fake)`. Attributed as matching Rinberg et al. (2025), https://arxiv.org/abs/2511.02620.

4. **Inference units are isolated and boot-checked.** Weights are whitelisted at boot; the trace can therefore reference a model by a stable whitelist identity rather than embedding weights. Between boots, weights are assumed fixed.

5. **Periodic memory wipes via "forced memorization"** mean the inference unit's state (e.g. KV caches, prefix caches) is periodically destroyed; replayable packets cannot depend on long-lived cache state that the verifier cannot reconstruct. Combined with pool-level random sampling (§2.6), a packet should be self-contained.

6. **Privacy tiers constrain retention of traces.** Three inference offerings during the deal: Total ZDR (100% deletion, 100K H100e fleet cap, no finetuning API, "very aggressive classifiers"), Probabilistic ZDR (99.9% deletion, random samples "go to classifiers and to transparent public database if flagged as high-risk," 100M H100e cap), No ZDR (100B H100e cap, "compression / distillation mitigations"). A trace/evidence pipeline needs a retention/deletion policy that is itself auditable.

7. **Output compression and monitoring sit on top of correctness verification:** "there needs to not only be correct output verification, but also some sufficient combination of refusals, monitoring, and compression." "Never show any reasoning intermediates to users by default." Passive monitors may run over traces — so traces are consumed by more than the recomputation check.

8. **Evaluations are also verified:** "For workloads that consist of final training runs there can be an extra step that involves running evaluations that also get verified for correctness." Eval harnesses should emit the same packet format.

9. **Training packets are a target too.** Same mechanism, "network taps being used as evidence collection, and partial recomputation being used to verify correctness of the packets that make up the workload." A packet abstraction that works for a forward pass should extend to a gradient step (input batch → gradient update → checkpoint delta).

10. **Hardware and software standardization for reproducibility is expected**, as are formally verified stacks later: "Completely rewritten software stacks to be significantly or entirely formally verified." A determinism envelope should be specified precisely enough to become a standard (which kernels, reduction orders, dtypes, versions) rather than a property of one vendor's build.

11. **Timeline pressure:** the page wants a "minimal verification package" ready before a deal; inference-only verification gets ~80% of pre-deal R&D. "Several pilots ... improvements to the designs are iterated on" in 2027 — i.e. it expects working open-source prototypes soon.

---

## 6. Linked prior work (from the page's outbound links)

- Thinking Machines, *Defeating Nondeterminism in LLM Inference* — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (cited as the example of reproducibility work)
- arXiv 2511.20621 — https://arxiv.org/pdf/2511.20621 (second reproducibility citation)
- Rinberg et al. (2025) — https://arxiv.org/abs/2511.02620 (detection-by-sampling math)
- Baker et al. — https://arxiv.org/abs/2507.15916
- Scher et al. — https://arxiv.org/abs/2506.15867
- Harack et al., *Verification for International AI Governance* — https://aigi.ox.ac.uk/publications/verification-for-international-ai-governance/
- *Fingerprinting All AI Cluster I/O* (Oxford AIGI) — https://aigi.ox.ac.uk/wp-content/uploads/2026/04/Fingerprinting_All_AI_Cluster_IO.pdf
- Cankaya, *Tech note: network taps for AI verification* — https://amodo.substack.com/p/tech-note-network-taps-for-ai-verification
- Cankaya, *Catching misreporting about ML hardware* — https://nacicankaya.substack.com/p/catching-misreporting-about-ml-hardware-bd2
- MIRI TechGov, *Suppressing side channels in an untrusted data center via retrofitted defenses* — https://techgov.intelligence.org/blog/suppressing-side-channels-in-an-untrusted-data-center-via-retrofitted-defenses
- RAND: https://www.rand.org/pubs/research_reports/RRA2849-1.html and https://www.rand.org/content/dam/rand/pubs/working_papers/WRA3000/WRA3056-1/RAND_WRA3056-1.pdf
- Taalas, *The path to ubiquitous AI* (hardcoded weights) — https://taalas.com/the-path-to-ubiquitous-ai/

---

## 7. Distilled implications for our trace format and determinism envelope

Not on the page; my synthesis of the above.

1. **Packet = (inputs, instructions, seed, outputs), self-contained and independently replayable** on foreign hardware. "Instructions" must pin the model (whitelist identity) and enough of the stack to reproduce exactly; the page leaves that encoding to us.
2. **Seeds are an input, not an output**, and must be acceptable from an external party. Sampling must be a pure function of (logits, seed).
3. **Exact (bitwise) token reproduction across machines** is the implicit bar; there is no tolerance in the math. Reproducing logits is a means, not a stated requirement — but it is the only practical way to make token equality robust, and it is what makes a mismatch diagnosable.
4. **The operator is the adversary.** No operator-signed hash substitutes for re-execution. Anything the verifier needs must be capturable passively from the wire or fixed at approval time.
5. **Granularity must be variable** (request → forward pass → sub-step) with the same schema; sampling is over a pool, so any single packet must be replayable in isolation.
6. **Multi-GPU within an inference unit is in scope; cross-unit scale-out is not** (it is physically removed).
7. **Overhead budget is soft but real**; the verifier's recompute budget is ~1%, and reproducibility overhead on the operator side is accepted only if "relatively small."
8. **Carry dtype/number-format and stack identity** because the plan expects both to change over the deal.
9. **Retention policy is part of the design** because of the ZDR tiers.
