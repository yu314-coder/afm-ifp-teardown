# The sparse IFP 3B, from the live asset

*2026-08-10.* Teardown of `afmplus-v11.1-ifp` — the model the on-device inference service actually
loads (`56659e51…`, confirmed by `lsof` on `TGOnDeviceInferenceProviderService`).

**No Apple weights or tokenizer data are in this repository.** What follows is the decode procedure
and the evidence for it. Everything here is reproducible against your own device's asset.

---

## 1. The expert pool was never missing

This retracts the central claim of the earlier teardown.

`ifp/ifp_rasterized_weights.bin` is **self-describing**. A 0x60 header, read literally:

```
magic  'BBBB'
u32[1] = 0x017C0100  ->  n_experts 380, expert_size 256
u32[2] = 0x0600002C  ->  hidden 1536,   n_layers 44
u32[3] = 3           ->  n_ffns (gate/up/down)
palette @ +80        ->  fp16 [-1.5, -0.5, +0.5, +1.5]     <- 2-BIT, not 4-bit
secA @ 96        135,168 B  = 44 x 1536 fp16        down scales, layer-shared
secB @ 135,264 17,121,280 B = 44 x 380 x 2 x 256    gate/up scales, per expert
payload @ 17,268,736  4,930,928,640 B
```

Closes to the byte three ways: `44*380*3*256*1536/4 = 4,930,928,640`, and
`17,268,736 + 4,930,928,640 = 4,948,197,376` = the file size exactly.

**380 experts per layer ship — the complete pool.** Not 46, not 219. Addressing is pure affine:

```
off(L, e, slot) = 17,268,736 + L*112,066,560 + (e*3 + slot)*98,304     slot 0/1/2 = gate/up/down
```

### Why the earlier "data-availability wall" was wrong

Prior work decoded this file as **4-bit**. It is 2-bit. That single error is sufficient to produce
the exact symptom three earlier workflows chased for ~2.1M tokens: gate/up partially decoding while
down read as pure noise at R≈1.0. The wall was a codec artifact, not a property of the asset.

`ifp_constant_table_ifp{1,2,3}_r48.json`, which selects *which* 46 of the 380 are resident, is not
shipped — but that is a runtime decision re-made every 32 tokens, and because the FFN is an
**ungated, permutation-invariant sum** the *order* within a chosen set is a non-issue. It is also
not what blocks reconstruction: swapping to a disjoint 46 moves normalized perplexity only
1.411 -> 1.481. **See §6 for what is actually missing.**

---

## 2. The ANE output axis is striped, not blocked

The load-bearing finding, and the one that unblocked the backbone.

Attention and dense-FFN weights are **not** in `main-h16g.odix` (that file is embeddings + RoPE, now
fully accounted). They are in `binary_0.hwx`, segments `__KERN_0..3`, in the same 2-bit container.

Every projection is split across **16 ANE units**. The natural assumption — unit `t` owns the
contiguous slice `[t·OUT/16, (t+1)·OUT/16)` — is wrong. Units are interleaved **block-cyclically at
bank granularity**: consecutive 16-row banks go round-robin to units 0..15.

```
assembled (wrong)  a = t*(N/16) + 16*b + l
TRUE               m = 256*b    + 16*t + l          t = unit, b = bank slot, l = lane
fix                T.reshape(16, N//256, 16, IN).transpose(1,0,2,3).reshape(N, IN)
```

Only **OUT** axes are affected. Every IN axis was always correct, because each unit holds the full
input width so IN never passes through the unit split. That asymmetry — IN passing at ~12× while OUT
fails — is the signature to look for.

### Why it resisted so long

A stripe is **not a block permutation**, so no 16-block or 96-block assignment search can express
it; those negative results were correct but aimed at the wrong family. Row-level stride-16
(`m = 256b + 16l + t`) is also wrong (0.021 = noise). The granularity is 16 rows, not 1.

### Evidence

Two independent investigations, in separate working directories, converged on the identical formula;
their corrected weight files are **218/218 bit-identical**. Five mutually independent instruments,
each per-layer against a permuted-ruler null:

| instrument | before | after |
|---|---|---|
| dense `gate~down` (internal SwiGLU alignment) | −0.4 sd, 0/12 layers | **+47.8 sd, 12/12** |
| dense `up~down` | 1.2 sd, 0/12 | **+43.9 sd, 11/12** |
| `o_proj` vs odix-embedding ruler | 1.72×, 1/56 | **7.39×, 34/56** |
| **cross-file**: `o_proj` vs the IFP file's own OUT profile | 0/44 | **41/44 at 0.2638** |
| RoPE lag-64 on Q | +0.796, residual **−0.16** (artifact) | **+0.907, residual +0.827, 56/56** |
| free Hungarian (unit σ, lane τ, bank order) | — | exact identity, independently |

The RoPE line is worth dwelling on: in the broken frame it read +0.796, which looks like strong
confirmation. Its residual after removing lane and block main effects was *negative*. A wrong frame
can manufacture plausible evidence, so a headline correlation without a residual check is not proof.

### Consequences

- **`Q ++ K ++ V`**, not `Q ++ V ++ K` (2048/1024/1024). Rows 2048:3072 carry the RoPE spike (+0.78),
  3072:4096 do not (−0.000); the OV circuit couples `o_proj`'s IN columns to 3072:4096 at +0.405
  (30/35 layers). The opposite call had been made in the broken frame.
- **`n_kv_head` = 8**, not 4 — forced by 4096 = 2048 + 1024 + 1024.
- Both originally-suspected causes were red herrings: `_ne_t` really is the unit index (327/327
  chunk-sets, symbol-level), and the "bigger DMA chunk holds the lower banks" rule was already right.

---

## 3. The norm gammas are present, and anonymous

Also a retraction: earlier work concluded gammas were "folded into the ANE coefficients, not
isolable statically." They are explicit.

They live in `binary_0.hwx` → `__TEXT` / `__const` — a 55 MB section, table at file offset
`0x0aabc000`, 717,312 B. The constants are **anonymous**: their symbol names are
`__<region>_<u64 content-hash>`, which is why every name-based search failed. The table repeats
byte-identically 10×, once per specialised ANE function.

```
for L = 0..55:
    qk_norm_query   [128]
    qk_norm_key     [128]      L = 0..33 only (vanishes where KV-reuse begins)
    attention_residual_connection_pre_residual   [1536]
    attention_residual_connection_post           [1536]
    feed_forward_residual_connection_pre_residual[1536]
    feed_forward_residual_connection_post        [1536]
output_norm [1536]
```

Period per two-QK layer: **6400 fp16 = 4·1536 + 2·128**. Two independent decodes — one via the
symbol table, one via pure statistical periodicity with no symbols at all — agree on **316/316**
tensors. Role names are confirmed against the declaration order in the less-stripped v11.0 build.

### Two families, applied differently

**Superseded by §5.** An earlier reading assigned these four slots as
(attention pre-residual, attention post, ffn pre-residual, ffn post) with a mixed `1+w` / `w`
convention. The ANE task stream settles it instead: the four `[1536]` gammas of a block are
`hn0` = attention branch-output norm, `hn1` = the post-add norm that *replaces* the stream and feeds
the FFN, `hn2` = FFN branch-output norm, `hn3` = the post-add norm feeding layer L+1. **All four are
applied directly (`w`), never as `1+w`** — the V ruler independently rejects `1+w`, which clips
29.1% of entries on 34/35 layers where `w` clips none.

Addressing: the constants are one flat address-ordered stream `x[0..226]`; `x[0]` is a leading
non-gamma vector, `x[1..224]` the 224 = 56x4 per-layer gammas, `x[225]` the output norm, `x[226]`
zero pad. `gamma(role, L) = x[4L + a]`.

### The architecture GGUF cannot express

There are **four** `[1536]` norms per layer plus QK norms — a `NormalizedResidualConnection` that
normalizes both the branch output *and* the post-add stream. GGUF's `attn_norm`/`ffn_norm`/`q_norm`/
`k_norm` slots leave **112 tensors with no home**, and llama.cpp's pre-norm template is the wrong
dataflow. The dense 3.33B sibling has the same structure. This is a *structural* deviation, not a
missing-value one.

---

## 4. A method note: diagnosing a reconstruction by its failure signature

When a reconstructed model produces garbage, the *shape* of the garbage identifies the defect class.
Ablation on a known-good model of the same architecture calibrates the scale.

Normalized perplexity `ln PPL / ln V` on an identical corpus (1.0 = chance):

| condition (on a known-good 4B) | normalized |
|---|---|
| baseline | 0.035 |
| `output_norm` = 1 only | 0.165 |
| all hidden norms = 1 | 0.993 |
| QK-norms = 1 only | 1.074 |
| **all norms = 1 — ceiling of gamma-only damage** | **1.124** |
| KV byte-copy across 15 layers (real gammas) | 0.521 |
| **weight rows stripe-permuted (real gammas)** | **1.36 – 1.42** |

Two distinguishable signatures:

- **Missing gammas → a flat distribution.** Top-20 holds under 1% of the mass, p1/p2 ≈ 1.2, the
  argmax is a coin-flip among ~150k near-ties, and output stays prompt-conditioned. "Correct but
  unnormalized."
- **Structural error → a sharp distribution.** Confident, well-formed, and input-independent.
  Worse than chance, because the model is confidently wrong rather than uninformed.

Per parameter, QK-norms dominate: ablating 72 QK tensors (9,216 numbers) does more damage than all
73 hidden norms (186,880 numbers). Broken attention has its own tell — the model echoes its input.

The practical lesson: *identify the defect class before hunting for data*. A missing-value hypothesis
and a structural hypothesis predict opposite distributions, and the test costs one ablation run.

---

## 5. The architecture is POST-NORM — the stream is replaced, not accumulated

*Supersedes the dataflow given in the first version of this section.*

```
h[L+1] = RMSNorm( RMSNorm( h[L] + RMSNorm(attn(h[L]))*hn0[L] )*hn1[L]
                  + RMSNorm(ffn(.))*hn2[L] )*hn3[L]
```

**There is no unnormalised running sum, and neither branch has a pre-norm.** The post-add norm
*replaces* the stream: `hn1[L]`'s output is simultaneously the FFN's input and the FFN add's stream
operand; `hn3[L]`'s output is both layer L+1's `q_proj` input and its attention add's operand. The
branches read the stream directly.

Every transformer reconstruction instinct says `h = h + f(norm(h))`. This model is `h = norm(h + f(h))`.

### The evidence

Read from L2 buffer **base** addresses in the ANE task stream (an earlier pass read the *Cfg* words
instead of the *Base* words and wrongly concluded addresses "don't survive the add"). Task 3326
(`ADD#2`) has `Src2Base = 0x2e000` — the buffer written by the `hn1` gamma-multiply and the `Src1`
of all twelve FFN projections. Tally over six regions, with ambiguous readings counted as
undetermined rather than as support:

| | replaces-stream | accumulates |
|---|---|---|
| ADD#1 | **357** | 0 |
| ADD#2 | **285** | 0 |

Independently, the unnormalised sum is *destroyed* before ADD#2 runs (two tasks overwrite both
copies and no DRAM write-out is armed), so the accumulating form is not merely unsupported but
impossible.

Branch order is **attention before FFN**, proven from matmul shapes in a middle layer:
`attn -> o_proj 2048->1536 -> norm*hn0 -> ADD#1 -> norm*hn1 -> FFN (gate/up 1536->5888 x2 chunks,
SwiGLU, down 5888->1536) -> norm*hn2 -> ADD#2 -> norm*hn3 -> q_proj of layer L+1`.

Norm counts match exactly: 4x[1536] + 2x[128] for KV-owning layers, 4x[1536] + 1x[128] for the 21
KV-reuse layers. At the head, `RMSNorm*hn3[55]` and `RMSNorm*output_norm` run back to back.

### Why this matters more than the numbers suggest

Implementing it makes the measured scores **worse** (dense layers 1246 -> 6724; 56 layers
11283 -> 116074). That is the point. Under an accumulating residual a growing unnormalised stream
dilutes every branch error, so a broken model degrades gracefully toward the identity — **the wrong
architecture was masking defects**. Under the real one each half-layer fully rewrites the stream and
errors propagate at full strength. It is a far sharper instrument, and it is what the silicon runs.

Two self-consistency checks came out right without being tuned for: the embedding's absolute scale
becomes **exactly irrelevant** (identical results at x1, x9, x32), which is required since the first
stream-replacing norm wipes it; and **both branches become productive** (removing the FFN gives
143312, removing attention 74878, against 9280 for the full model), where under the accumulating
form the FFN had been neutral-to-harmful.

### Two clean negatives

`specialized_model_0.mpsgraph` contains **no arithmetic graph at all** — 7 op names, all 108 ANE ops
external with empty regions, 99.9% of 2.4M attributes are location metadata. Strings like
`ResidualAdd_743` are `NameLoc` labels, i.e. *order*, not edges; no SSA edge is recoverable from it.
And `x[0]`, the leading vector of the gamma stream, is **not a norm gamma of any class**
(767/1536 entries negative), so using it as layer 0's input norm is unsupported.

### What GGUF can and cannot hold

The best achievable mapping into the `gemma3` architecture is
`attn_norm <- hn3[L-1]`, `post_attention_norm <- hn0[L]`, `ffn_norm <- hn1[L]`,
`post_ffw_norm <- hn2[L]` — which the published build already carries. The irreducible gap is the
residual operand: gemma3 accumulates onto the *unnormed* stream, AFM onto the normed,
stream-replacing value. No GGUF architecture expresses that.

---

## 6. The op order, read off the graph

Not guessed. The MLIR string section of `specialized_model_0.mpsgraph` interns strings in IR-walk
order, so the sequence *is* the op order. Per layer, verbatim:

```
convolution(o_proj) -> [ANE_RMSNorm | NormalizedResidualConnection] mul mean add sqrt div mul
                    -> add [ResidualAdd] -> mul mean add sqrt div mul -> next branch
```

so

```
h <- h + post_gamma * RMSNorm( branch( pre_gamma * RMSNorm(h) ) )
```

Attention internals, in graph order:

```
convolution(fused 4096) -> split_with_sizes(2048,1024,1024) -> [ANE_RoPETransform] -> [ANE_QKNorm]
   -> KVQuantizer -> KVCacher -> ANE_ScaledDotProductAttention -> o_proj
```

**RoPE is applied BEFORE the per-head QK-norm**, against the usual convention. (It barely matters
numerically here because gamma[d] ~ gamma[d+64] at r = +0.91; flipping the order moves NLL by
< 0.01. But it is what the graph says.)

Op counts corroborate the gamma extraction exactly: `mean`/`sqrt`/`div` each appear **315** times =
224 residual norms + 1 output norm + 56 q-norms + 34 k-norms, and `neg` appears **91** times = 56
Q-RoPE + 35 K-RoPE. Both match the 316 recovered tensors and the 35 KV-owning layers.

Settled the same way: `linear_0` takes the SiLU and `linear_1` is the ungated branch; the 21
`kv_reuse` layers all read one shared K/V pair produced by layer 34; the RMSNorm epsilon and mean
divisor are explicit registers in the ANE task descriptors; the 23 shared experts are stored
experts 0..22.

---

## 7. Why it does not work

A standalone forward pass implementing the real dataflow (which no GGUF architecture can express)
gives, on 219 next-token targets with V = 262144 (chance NLL 12.4766):

| N layers | 0 | 1 | 2 | 8 | 24 | 56 |
|---|---|---|---|---|---|---|
| median true-token rank | 1307 | 2381 | 6149 | 10528 | 23192 | 74380 |

With **no** layers at all — embedding, output norm, tied head — self-top-1 is 1.000 and the median
next-token rank is already 1307, so the embedding and head are mutually consistent and the raw
embedding carries real next-token signal. From layer 1 onward the rank degrades **monotonically**,
and layer 1 alone is *not* at chance. There is no single broken component: every layer degrades a
little. Meanwhile the residual stream's token-to-token cosine climbs 0.375 -> 0.980, collapsing onto
one shared direction — which is precisely why the output becomes input-independent.

That is the signature of **uniformly distributed weight error**, not of a miswired or permuted
tensor family.

### The cause: the accuracy-recovery adapter ships untrained

The base weights are plain **round-to-nearest 2-bit** — the code histogram matches RTN-of-Gaussian
at the MSE-optimal scale to within 0.003, i.e. roughly **34% relative weight error**. Apple's
palettization scheme compensates for that with a **rank-48 accuracy-recovery LoRA**. That adapter is
shipped as an untrained placeholder.

Verified two independent ways:

* At LoRA-factor granularity the rms distribution is sharply **bimodal and alternating** — 48.8% of
  blocks sit at ~1.1e-5 (still at initialization) interleaved with blocks at ~2e-2. One factor of
  every A/B pair was never trained, so the product is ~0 and the adapter is a no-op.
* Apple's own `metadata.json` gives the adapter signature as
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` — the SHA-256 of the **empty
  string**.

So the shipped asset contains a 2-bit base model and an empty corrector — all four `lora_*` blobs
show the same half-untrained pattern (48.8-51.0% of blocks still at initialization).

### But this cannot be the whole explanation, and the objection is decisive

**The device runs this exact asset and produces coherent text.** A live core dump caught
`TGOnDeviceInferenceProviderService` mid-generation, writing a coherent essay, with this same
`56659e51` file open. If these bytes are sufficient on-device, they are sufficient in principle, and
an absent corrector cannot be what stops us.

The likelier reading is that the model is **quantization-aware-trained**: 2-bit is the intended
representation and needs no recovery adapter, the shipped `lora_*` slots are task-adapter mounts
that are legitimately empty for the base model, and the "34% relative error" figure measures the
distance to an fp16 original that never existed. Under that reading the untrained adapters are
expected rather than damning, and **some defect remains in this reconstruction that we have not
found.**

What survives regardless: the layer sweep shows uniformly distributed error rather than one
miswired family, so whatever remains is diffuse — a systematic per-layer discrepancy, not a
permuted tensor. Candidates not yet excluded: the dequantization scale convention (per-row scale
applied on the wrong axis, or a missing global factor), the RMSNorm epsilon and mean divisor as
actually implemented in the ANE registers, and the fp16-vs-fp32 accumulation the hardware uses.

This section states a hypothesis that its own strongest control contradicts. It is recorded that way
deliberately.

### A retraction: normalized perplexity cannot diagnose ordering

An earlier reading of this teardown placed the export "in the permutation band" (normalized
`ln PPL / ln V` of 1.41, against 1.36-1.42 produced by deliberately permuting weight rows). **That
inference was invalid.** A direct control refutes it:

| condition, on a known-good Qwen3-4B | normalized |
|---|---|
| the AFM 2-bit codec alone | **1.773** |
| a deliberate stripe permutation alone | 1.402 |
| codec **and** permutation together | **1.375** |

Damage is **non-monotone** in this range — adding a permutation on top of the codec *improves* the
number. Normalized perplexity between roughly 1.3 and 1.8 therefore carries no information about
ordering, and no conclusion about permutations may be drawn from it.

---

## 8. Honest status

**Solved:** the container and 2-bit codec (byte-exact); the complete 380-expert pool; the embedding
(`[262144,1536]` int4, per-token scales) and tokenizer; the FFN triple alignment for the 44 IFP
layers (identity, 94.3 sd); the striped OUT axis; `Q++K++V`; `n_kv_head` = 8; the 316 norm gammas
and how to apply them.

Also solved since: the exact dataflow and op order; `linear_0` takes the SiLU; the `kv_reuse` source
is layer 34; the RMSNorm epsilon and axis; the 23 shared experts are stored experts 0..22.

**Open:** which 46 of 380 experts are resident per *instruction* (a runtime decision, re-made every
32 tokens), and the sandwich-norm dataflow has no GGUF representation.

**The remaining defect is unidentified, and is diffuse rather than structural.** The layer sweep
shows uniformly distributed error, not one miswired family. All shipped `lora_*` adapters are
untrained no-ops, but that cannot be the cause, because the device produces coherent text from this
same asset (see §6) — so the 2-bit weights are sufficient in principle and something in this
reconstruction is still wrong. The exported GGUF loads and runs and does not produce coherent text.
It is published as a reproducible decode, not as a working model.
