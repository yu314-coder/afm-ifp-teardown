# PICO (afmplus-v11.0-pico) — NORM RECIPE for a runnable from-weights forward

Date: 2026-07-18
Asset: `…/purpose_auto/031c7be6…asset/.AssetData/model.odixpackage/` (verified `model_config: v11-pico`).
Files read: `program.odix` (136 MB), `MPSGraph/…/specialized_model_0.mpsgraph` (2.53 MB, MLIR bytecode
v6, producer `MLIR22.0.0git`), `binary_0.hwx` (193.9 MB ANE program), `manifest.plist`,
`program.dbginfo`, `lora_{32,64}_constant_data.bin`. No Apple weights committed.

---

## BOTTOM LINE — the exact norm recipe

Pico is **pre-norm RMSNorm + per-head QK-norm + rotate-half RoPE**, `sub`-free (no LayerNorm anywhere).
For a from-weights forward, **use unit gamma everywhere** — pico ships **NO explicit `[1024]` or `[64]`
gamma vectors** in any reachable artifact:

```
# per layer (24 layers, D=1024, GQA 16 Q-heads / 4 KV-heads, head_dim=64):
xn      = x * rsqrt(mean(x*x, axis=-1) + eps)          # input RMSNorm  (gamma=1, folded)
q,k,v   = conv1x1(xn)                                    # QKV proj -> Q[16,64], K[4,64], V[4,64]
q       = q * rsqrt(mean(q*q, axis=head_dim) + eps)     # QK-norm on Q, per head over 64  (gamma=1, TRUE)
k       = k * rsqrt(mean(k*k, axis=head_dim) + eps)     # QK-norm on K, per head over 64  (gamma=1, TRUE)
q,k     = rope(q), rope(k)                               # rotate-half, HD/2=32 pairs
a       = sdpa(q,k,v)  (causal, GQA repeat 16/4)
h       = x + conv1x1_O(a)                               # residual
hn      = h * rsqrt(mean(h*h, axis=-1) + eps)           # post-attention RMSNorm (gamma=1, folded)
g,u     = conv1x1_gate(hn), conv1x1_up(hn)
h       = h + conv1x1_down( silu(g) * u )                # SwiGLU + residual
```

- **RMSNorm, not LayerNorm** — `sub` op count is **0** across the whole graph (no mean-centering).
- **gamma = 1 at runtime, everywhere.** No gamma vector ships. See proof below.
- **eps** could not be extracted as a clean literal (baked into the ANE kernel); it is a small additive
  constant (~1e-5…1e-6 by AFM/Llama convention) with negligible effect on O(1–10) RMS values.
- **RoPE theta**: **not stored as a literal in any pico file.** cos/sin are precomputed host-side by the
  GenerativeModels runtime and fed as a Float16 `[1, ctx, 1, 64]` graph input; the base lives in the
  runtime, not the shipped model. Best value is the AFM-family **500000** (carried over from the 3B), but
  it is **NOT independently confirmable from pico's asset** — I explicitly did **not** find 500000 (or
  10000/100000/1000000) as fp32/fp64/inv-freq bytes anywhere.

---

## What is PROVEN (independently, this pass)

### 1. It is RMSNorm; the per-layer norm census is exact
The MLIR string table (op result / location names) gives clean, contiguous primitive-op ranges:

| op | count | range | meaning |
|---|---|---|---|
| `mean` | **96** | `mean_1..96` | reduce-mean(x²) — one per normalization |
| `rsqrt` | **96** | `rsqrt_1..96` | one per normalization |
| `sub` | **0** | — | **no mean-centering ⇒ RMSNorm, not LayerNorm** |
| `neg` | **48** | `neg_1..48` | rotate-half negate — one per RoPE application |
| `select` | 48 | — | RoPE half-select |
| `slice` | 96 | — | RoPE two-half slices (2 × 48) |
| `cat` | 95 | — | RoPE rotate-half concat + KV cache cat |

**96 = 24 layers × 4 normalizations = 2 hidden RMSNorms + q_norm + k_norm per layer.**
**48 = 24 layers × 2 RoPE (Q, K).** These three counts are mutually consistent and deduped to one
logical forward (the file's 18 shape-specializations share the primitive ops).

Composite location tags corroborate the roles: `ANE_RMSNorm`, `ANE_QKNorm`, `ANE_RoPETransform`,
`ANE_ScaledDotProductAttention`, `SwiGLU`, plus `IsolatedGatherPositionalEmbedding` /
`ANE_RotaryPositionalEmbedding` (odix). (Their raw string counts — 142/46/46/45/43 — mix specialization
variants and must not be read as layer counts; the primitive `mean`/`rsqrt`/`neg` counts are the reliable
ones.)

### 2. No explicit gamma ships — gamma = 1 at runtime
Checked every place a learned scale could hide; all negative:

| source | result |
|---|---|
| mpsgraph AttrType (2.1 MB, where dense constants live; Resource section is **empty**) | **no clean fp16 run reaches 1024** (max 733) and **no set of ~48 clean length-64 arrays** ⇒ no `[1024]` hidden-gamma, no `[64]` QK-gamma embedded |
| `binary_0.hwx` symbol table (29,991 syms) | only `K<hash>_ne_*` weight tiles + `rt_op_*` runtime ops; **zero** gamma/`norm.weight`-named or small-vector constants |
| hwx host-visible `__INIT`/`__RUNTIME` | max clean fp16 run 6–9 ⇒ no gamma tables |
| graph inputs (8 tensors) | RoPE tables, KV cache, mask, hidden, position — **no gamma-shaped input** |
| `lora_{32,64}_constant_data.bin` | LoRA adapter deltas only |
| `program.odix` module names | norm submodule names stripped; only `_wrapped_model…qkv_transform` (LoRA) survive; **no `…norm…weight` / `scale` / `gamma` leaf** |

- **Hidden RMSNorms (2/layer):** gamma is **folded into the adjacent linear at ANE-compile** — identical
  to the 3B (documented in `afm-forward-plumbing-diagnosis`; folding is on: `disableShapeFolding=false`).
  Whether the source had a γ is immaterial and unrecoverable separately: the recovered Q/K/V and gate/up
  conv weights **already contain it**, so a from-weights forward multiplies by nothing extra (γ=1).
- **QK-norm (q_norm, k_norm):** γ = 1 is a **stronger, positive result**, not just "folded". A QK-norm γ
  is provably **non-foldable** — RoPE (a rotation that mixes dims i and i+32, fed by *runtime* cos/sin
  inputs, not compile-time constants) sits between the norm and the next linear, and RMS-division is
  nonlinear, so a per-dim γ[64] could neither fold forward into RoPE/SDPA nor backward into the QKV conv.
  If it existed it would **have to** appear as an explicit `[64]` constant multiply. It does not exist
  anywhere (row 1–2 above) ⇒ **pico's QK-norm is genuinely parameter-free (unit γ).**

### 3. Per-head QK-norm IS present (definitive YES)
`ANE_QKNorm` composite + the op accounting: `q_norm` and `k_norm` each contribute exactly **one**
`mean`+`rsqrt` per layer (the reduction is over `head_dim=64` with the head axis as batch), i.e. 48 of the
96 normalizations are QK-norms. Applied to **Q (16 heads × 64)** and **K (4 heads × 64)**; V is not
normalized. Head structure confirmed from the qkv-adapter shapes in `manifest.plist`
(Q: 1024→1024 = 16×64; K/V: 1024→256 = 4×64). This mirrors the 3B (QK-norm over HD=128); pico is HD=64.

### 4. RoPE = rotate-half over HD/2 = 32 pairs, host-precomputed table
`neg=48`, `slice=96`, `cat=95`, plus `ANE_RoPETransform` / `IsolatedGatherPositionalEmbedding` — the
standard slice/slice/neg/cat rotate-half pattern, one apply on Q and one on K per layer. The cos/sin table
is a Float16 `[1, ctx, 1, 64]` **graph input** (manifest `inputShapes`; ctx = 1024 or 4096 across the two
entry functions), gathered by position. **Head_dim = 64.**

---

## NOT resolved / honest caveats

- **RoPE theta (exact base): OPEN.** Not a literal in `program.odix`, `specialized_model_0.mpsgraph`,
  `binary_0.hwx`, `program.dbginfo`, or the constant_data files — searched fp32/fp64 and the inv_freq[1]
  ratio for {10000, 100000, 500000, 1000000}: **0 hits**. The table is runtime-generated, so the base is
  in the GenerativeModels framework, not the model. **500000** is the structurally-consistent AFM-family
  value (same status as the 3B: "assumed, not independently re-derived"). To pin it exactly you'd fit
  `cos(pos·θ^(−2i/64))` to a captured RoPE-table input — a **non-privileged** runtime capture of that
  fp16 input tensor (no sudo needed), which is the clean way to close this.
- **eps (exact value): OPEN but negligible.** Baked into ANE kernel math; not a clean extractable literal.
  Use ~1e-5 (or 1e-6); it does not affect coherence.
- **Final pre-unembed norm:** **not** among the 24×4 = 96 ANE norms. It is applied host-side (the
  embed/unembed are host-side per project state) or folded into the LM head; treat it as a separate
  final RMSNorm whose γ, if any, is folded into the recovered/captured unembed. Not an ANE-graph object.
- **Pre-norm vs sandwich:** the 96 = 24×4 accounting (2 hidden + q + k) fixes pico as **pre-norm** (input
  norm + post-attention norm), i.e. **2 hidden RMSNorms/layer, not the 4-norm "sandwich"** the 3B notes
  mention. The exact residual placement (norm-then-sublayer, standard pre-norm) is the AFM/Llama
  convention; op counts confirm the count, not the wiring order, which is taken as the standard convention.
- **QK-norm-before-RoPE ordering** is the near-universal convention (Gemma2/Qwen/OLMo/AFM) and is
  consistent with the graph, but the strict op order is not separately proven from string counts.

---

## Comparison to the 3B (same AFM v11 family)
`afm_odix/attention_validation.json`: 3B = "input RMSNorm, QK-norm, RoPE theta=500000, GQA, causal SDPA,
o-proj residual", with theta=500000 flagged "assumed, NOT independently re-derived" and "not a literal in
main-h16g.odix". Pico reproduces this recipe exactly at its own dims (D=1024, HD=64, QK-norm[64]) and the
same theta-is-not-a-literal situation — so the family recipe transfers; only the scalar base stays a prior.

## Provenance
All findings reproduced this pass from the on-disk asset via `numpy`/`struct` (MLIR-bytecode
prefix-varint section walk; hwx LC_SYMTAB census; AttrType/host-section constant scans; manifest parse).
Scratch scripts under `/Volumes/D/fix/pico_shapes/`.
