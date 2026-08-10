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

The one thing genuinely absent is `ifp_constant_table_ifp{1,2,3}_r48.json`, which selects *which* 46
of the 380 are resident. Because the FFN is an **ungated, permutation-invariant sum**, the *order*
within a chosen set is a non-issue — only the *choice of set* is open.

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

| role | mean | %positive | application |
|---|---|---|---|
| attention pre-residual | 0.011 | 63% | **`1 + w`** — zero-centred, Gemma-style |
| qk_norm query / key | 1.61 | 99% | direct |
| attention post / ffn post | 0.67 / 0.056 | 100% | direct, and **small** |

The post-norms rise with depth (attention post: 0.11 → 1.10 across the stack) — a learned branch
damping, LayerScale-like, letting the residual stream dominate early. Applying `1+w` to those, or
raw `w` to the pre-residual norms, is as wrong as omitting them.

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

## 5. Honest status

**Solved:** the container and 2-bit codec (byte-exact); the complete 380-expert pool; the embedding
(`[262144,1536]` int4, per-token scales) and tokenizer; the FFN triple alignment for the 44 IFP
layers (identity, 94.3 sd); the striped OUT axis; `Q++K++V`; `n_kv_head` = 8; the 316 norm gammas
and how to apply them.

**Open:** which 46 of 380 experts are resident (the constant table is not shipped), so any export
must fabricate the selection; the sandwich-norm dataflow has no GGUF representation; `norm.eps` is
baked into the ANE kernel.

**The exported GGUF loads and runs but does not produce coherent text**, and its perplexity sits in
the structural band, not the missing-value band. It is published as a reproducible decode, not as a
working model.
