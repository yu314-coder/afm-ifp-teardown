# PICO (afmplus-v11.0-pico, 300M, H16) — RUNNABLE-FORWARD STATUS

Date: 2026-07-18
Synthesis of: `pico_norms.md`, `pico_head.md`, `pico_forward.py`,
`pico_arrangement_sweep*`, the consistency-signal suite
(`gauge_demo.py`/`gate_up_align.py`/`ffn_align.py`/`residual_align.py`/`qk_rope.py`/`signal_a_tied.py`),
`pico_canonical_arrangement.py`, and the prior verdicts
`PICO_WEIGHT_RESULT.md` / `PICO_ARRANGEMENT_RESULT.md` / `PICO_DOWNPROJ_RESULT.md`.
No Apple weights are committed.

---

## 1. Is there a RUNNABLE from-weights pico forward? — **PARTIAL**

**It runs, it is stable, and it is correctly wired — but it is NOT a validated-correct forward.**

What is solid (a real "runs from weights" milestone):

- **Norms resolved.** Pico is `sub`-free → **RMSNorm, not LayerNorm**. Clean op census `mean=rsqrt=96`,
  `neg=48` = 24 layers × (2 hidden RMSNorms + q_norm + k_norm) + 24×2 RoPE. Pre-norm (2 hidden norms/layer),
  **γ folded into the adjacent linear at ANE-compile** → a from-weights forward uses **γ = 1** and the
  weights as-is. No explicit γ vector exists anywhere reachable (mpsgraph AttrType, hwx symtab/`__INIT`,
  odix, `constant_data.bin`) — verified absent, not merely unfound.
- **Per-head QK-norm = unit γ, PROVEN two ways.** Present (`ANE_QKNorm`, head_dim 64), non-foldable (RoPE
  sits between norm and next linear), so a learned γ would have to ship as an explicit `[64]` constant — it
  appears nowhere. Independently reconfirmed numerically: QK-norm OFF collapses attention entropy to ~0.02
  (one-hot saturation); ON gives structured ~0.85.
- **Forward executes end-to-end.** `pico_forward.py` runs all **24 layers** (GQA 16Q/4KV/hd64,
  `repeat_interleave` fan-out, causal mask, RoPE, SwiGLU ffn=3200, residual stream), finite throughout, on
  the captured embedding rows (`tw_{dog,dogs,king,kings,London}.core`) at T=1 and T=8. Final post-norm
  |x| = √1024 = 32.0000 (correct RMSNorm sanity value). Weights come from the validated per-tile codebook
  decode (`pico_weights.decode_tensor`/`decode_down`), all 168 logical tensors.
- **Stable / bounded.** Across 48 arrangement variants the per-layer residual-RMS ratio decays to ~1.01 and
  plateaus — the pre-norm signature. No NaN/inf in any finite arrangement.

What is NOT yet true (why this is PARTIAL, not YES):

- **No functional (semantic) validation.** There is no comparison against a ground-truth pico output. A
  stable, finite, reproducible run under a stated convention is exactly that — it does **not** show the
  numbers are the model's true activations.
- **RoPE base θ is assumed, not pinned.** θ=500000 is the AFM-family carryover and is structurally
  consistent, but pico's asset contains no `{10000,100000,500000,1000000}` literal — cos/sin are a runtime
  `[1,ctx,1,64]` fp16 **graph input** computed by the GenerativeModels framework, so the base is not in the
  shipped model.
- **Residual magnitude is suspicious.** Decoded N-tensor weights are ~9× a typical trained scale
  (RMS ~0.28, absmax ~0.98, unnaturally unit-bounded) → a likely **missing per-group scale** (the tile
  header `[64:96]` fp16 slot, left out as spectrum-neutral). This drives a large layer-0 step and the
  monotone residual growth; it is identical across arrangements and, because QK-norm renormalizes q/k, does
  not corrupt the arrangement conclusions — but it must be chased for a semantically-faithful forward.
- **Down-proj seams + RoPE pairing unpinned** (see §2) — these *do* affect the output.

**Bottom line for §1:** the plumbing bar is cleared (norms resolved, 24-layer forward runs, stable,
correctly wired up to gauge). The *correct-output* bar is not. **PARTIAL.**

---

## 2. Arrangement verdict — **CONSTRAINED / CANONICAL-CONVENTION-APPLIED — NOT PROVEN**

Not "still-open" wholesale (it narrowed dramatically), and **not proven** — and explicitly **not proven by
stability**. Stability is a *necessary* filter every finite arrangement passes; a pre-norm net re-normalizes
each sublayer, so boundedness cannot, by itself, rank one arrangement over another. Any claim of proof from a
stable run is rejected here.

How the three signals narrowed it:

**(a) Consistency — the decisive narrowing (gauge theorem, numerically demonstrated in `gauge_demo.py`).**
A full forward is invariant to a *consistent* permutation of a logical axis; only *relative* cross-tensor
consistency matters. Demonstrated on the real decoded weights + captured embeddings (baseline |x|=71.55):
- consistent residual-axis permutation across {Q,K,V,gate,up}.Cin, {O,down}.Cout, and the input embedding →
  output identical up to that permutation (rel 1.2e-4 = float roundoff);
- consistent FFN-neuron permutation across {gate,up}.Cout and down.Cin → output unchanged (4.7e-4);
- **break** either consistency → output blows up to O(5).
- Tied embedding (`signal_a_tied.py`): input=output embedding, so the residual order is a gauge that
  **cancels end-to-end** (permuted tie rel 8e-5; broken tie rel 1.47).
- Q/K share head-dim order by construction (same N-codec): shared head-dim permutation is invariant (1.8e-4),
  Q-only breaks (1939).
- gate_i↔up_i neuron alignment corr **+0.934**, z≈+53 over all 24 layers — decisive, and it doubles as
  empirical proof the N-codec decode is neuron-self-consistent across two independent tensors.

→ The **entire residual-axis arrangement, the entire FFN-neuron arrangement, and the Q/K head-dim order** are
**gauge** (free, cancel). Essentially all geometry of the **144 N-codec tensors (Q/K/V/O/gate/up)** need NOT
be resolved for a correct forward. This is the big win: the wall shrinks from "byte-exact composition of 168
tensors" to a short list.

**(b) Numerical health — resolves two sub-knobs (weaker than proof, but agrees with independent evidence).**
- K/V must be read as `[1024,256]` row-major reshape (entropy 0.856, 24/24 plausible), not `[256,1024]`
  transpose (attention collapses to uniform ~0.98). Matches the map's declared shape.
- QK-norm ON (see §1). Both are non-degeneracy discriminations, consistent with prior evidence — not proofs.

**(c) Canonical convention — a primary-source element→position map, applied (but SV-invisible for pico).**
`pico_canonical_arrangement.py` cites the on-machine, unstripped **ANECompiler.framework** (disassembled):
`GetNumOutputChannelsPerCycle`→16 output lanes innermost; `GetShufflingOrder`→output z-order (bit-exact vs an
Apple gate/up ground truth); `GetDeInterleaveShufflingOrder`→intra-tile lane lift; OCG hard-capped at 16/pow2.
Cross-checked by an Apple-toolchain read-back of a `[1024,1024]` palettized conv (bijection over 1,048,576
positions). Applied to pico's 128×128 container it lands **91.4%** of layer-0 Q elements at a different home
than the old placeholder `.T`, with structure-R essentially unchanged (9.347 vs 9.178) — a **genuinely
distinct, standards-based arrangement**, re-confirming SV cannot rank it.

**Still genuinely OPEN — the narrow, precisely-characterized residual (and it *does* affect the output):**
1. **down-proj cross-codec seams** — down.Cin ↔ gate/up.Cout (FFN order) and down.Cout ↔ N-codec residual
   order. `down` is the sole L-codec tensor on both axes; no static coupling exists (in↔out neuron-norm
   corr ≈ 0, z −0.07). Not pinnable by any weight statistic.
2. **s half-block seam** inside gate/up (where the 128 s-neurons sit among 3200; small).
3. **RoPE absolute pairing** (interleaved vs rotate-half) + **GQA head grouping** — discrete runtime
   conventions, provably not in the weights (within-pair norm asymmetry indistinguishable from random;
   GQA subspace overlap ≈ chance). The proven "rotate-half" rests on the `neg=48`/`slice=96` op-counts, not
   on any weight fit; θ base lives in the framework.
4. The tile→cell / intra-128 lane order for the N-codec tensors is canonical-convention-applied but
   **SV-invisible for pico's specific 128×128 container → not file-proven** (as `PICO_ARRANGEMENT_RESULT.md`
   established).

**Bottom line for §2:** consistency **eliminates** the arrangement of 144/168 tensors as gauge and validates
the decode's self-consistency; health **constrains** two sub-knobs (K/V shape, QK-norm) in agreement with
independent evidence; the canonical convention **supplies** a primary-source element→position map for the
remaining N-codec geometry. The byte-exact arrangement is therefore **strongly CONSTRAINED and has a canonical
convention applied — but it is NOT PROVEN.** The irreducible open item collapses to the **down-proj
cross-codec seams + the RoPE runtime pairing** — items no static/non-privileged signal can rank. Only a
**functional forward-pass oracle** can close them.

---

## 3. The one definitive next step (operator, requires sudo)

Capture pico's real output for a known short prompt and use it as a functional oracle that ranks the residual
arrangement permutations by **argmax-match** (and by per-position activation match). Deliverable:

- **`/Volumes/D/fix/capture_pico_logits.sh`** — attach-by-pid to the live inference service (proven method),
  arm an auto-continue breakpoint on `_espresso_network_bind_buffer`, and dump the CPU-readable buffers for
  pico's graph I/O tensors **`in_embeddings`** (the exact `[T,1024]` input the forward consumes) and
  **`placeholder_out_opt_logits`** (the `[…,262144]` output) for a fixed `-t 0 -m 1` prompt; also record the
  greedy emitted token (the argmax anchor) and take a `save-core --style full` fallback.

Then, offline (no sudo): feed the captured `in_embeddings` through `pico_forward.py`, sweep the **only**
remaining non-gauge knobs — down.Cin↔ffn order, down.Cout↔residual order, the s-block seam, and
{rotate-half vs interleaved RoPE} × GQA grouping — and select the arrangement whose logits `argmax` matches
the captured greedy token (cross-checked by minimizing per-position activation error vs the captured logits).
That is the documented remaining wall's only non-privileged closer; everything upstream is already gauge or
canonical-convention-fixed.

**Honesty guardrail:** a stable, finite, self-consistent forward under the canonical convention is what exists
today. It is a real upgrade from "runs but wholly unvalidated" (144/168 tensors reduced to gauge), but it is
**not** a proven byte-exact arrangement and **not** a validated semantic forward. Both of those require the
oracle above.
