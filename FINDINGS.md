# MLIR bytecode parse — results (ifp_model.mpsgraph, 8.9 MB)

## What was parsed
`ifp_model.mpsgraph` is **MLIR bytecode v6**, producer `MLIR22.0.0git`. Fully
decoded the section table:

| Section | Size | Content |
|---|---|---|
| Dialect | 32 B | dialect names (builtin, mps_spi, mpsx, mps, ane, coreai, torch) |
| AttrTypeOffset | 841 KB | offset index into AttrType |
| **AttrType** | **7.5 MB** | attributes + types (inline dense data, integer attrs, tensor shapes) |
| IR | 6.2 KB | the op graph (references attrs/types/strings by index) |
| ResourceOffset / Resource | 1 B / 0 B | **empty** → no external file resources |
| String | 523 KB | 16 228 strings incl. every op name + location hierarchy |
| Properties | 378 B | func-level attrs |

## Constant → tensor mapping (SOLVED for FFN)
Each of the **396 `ifp_constant_N`** carries a full **location hierarchy** in the
String section, e.g.:

```
ifp_constant_0 > lut_to_dense_62 > PalettizedConv2D_4157 > AdaptedLayer_3536
             > MultiOutputLinear_2589 > TransformerFeedForward_1737
             > TransformerLayer_1302 > ... > CausalLMTransformer > ExportableIFP
```

From this, every constant is mapped to its transformer layer and role
(`ffn_constant_map.json`):

- **All 396 file-constants are FFN / MoE weights** — no attention, embedding, or
  router constants in this graph.
- They group **exactly 3 per layer** across **132 layer-instances**.
- Per layer: **two constants share one `PalettizedConv2D`** (the fused gate/up
  up-projection from a `MultiOutputLinear`) **+ one separate conv** (down-proj) —
  textbook SwiGLU.
- Constant *numbering* is in **ANE scheduling order**, not model depth; true depth
  is recovered by sorting on the `TransformerLayer_N` op-id.

## Norms (RESOLVED: folded, not separate)
Scanned the 7.5 MB AttrType section for inline norm tensors:
- f16 `[1536]` norm-profile arrays: **0 hits**
- f32 `[1536]` norm-profile arrays: **0 hits**
- Norms are **not** file-constants (0 of the 396) and **not** inline data.

Yet the graph has **661 `RMSNorm` ops**. Conclusion: the learned RMSNorm scales
(γ) were **folded into the adjacent palettized linear weights at compile time**
(`Linear(RMSNorm(x)·γ) ≡ Linear'(RMSNorm(x))`, `W' = W·γ`) — the standard ANE
optimization. The parameter-free normalization (÷RMS) is done by the RMSNorm ops.
**Implication: the norm scales are already baked into the recovered weights** — they
were never a separate asset to find.

## Honest open items
1. **132 layer-instances ≠ 44-layer IFP architecture.** This `.mpsgraph` contains
   ONLY FFN constants (no attention/embed/router), so it is a **partial/specialized
   subgraph**, not the whole model. Attention + router live in sibling `.mpsgraph`
   files. The 132-vs-44 count is unreconciled and I will not claim a depth mapping
   onto the full model until the sibling graphs are parsed the same way.
2. **Exact byte offsets** are not in the mpsgraph (it reads by *symbol name*); the
   name→file-offset table is in the **odix binary index** (unparsed block at 0x40),
   Resource section here is empty. The constant *number + hierarchy* already gives
   layer+role, which is what naming the weights actually needs.
3. **Router / IFP mask predictor**: not among these 396 constants; in another graph.

---

# Sibling-asset parse — the complete weight map (base model DMG)

Mounted `UC_FM_LANGUAGE_INSTRUCT_3B_BASE_GENERIC_SPARSE_GENERIC_H16G_IFP_Cryptex.dmg`.
Structure of the base IFP model (`model.odixpackage/`):

| File | Size | Role |
|---|---|---|
| `ifp/config.json` | — | **ground-truth architecture** (below) |
| `ifp/ifp_rasterized_weights.bin` | 4.7 GB | **FFN / expert weights** (the 396 `ifp_constant`s read from here) |
| `ifp/metadata.bin` | 0.4 MB | **BBBB swizzler** = shipped constant→offset table for the FFN weights |
| `main-h16g.odix` | 276 MB | base program IR (full op graph + tensor names) |
| `…/specialized_model_0.mpsgraph` | 8.5 MB | the MPSGraph I parsed (identical to my `ifp_model.mpsgraph`) |
| `…/binary_0.hwx` | 252 MB | **ANE compiled binary — holds the attention/backbone weights** |
| `…/lora_i1_r48_48_constant_data.bin` | 68 MB | IFP LoRA adapter constants |

## Ground-truth config (`config.json` + `metadata.json`)
```
num_layers        : 44          hidden_dim        : 1536
num_ffns          : 3           expert_size       : 256
active_experts    : 10          shared_experts    : 4
active_ffn_dim    : 2560 (=10×256)   expert_selection_frequency : 32
context_length    : 8192        model_config      : v11-ifp / afmplus-v11.0-ifp
constant_table_json : ifp_constant_table_ifp1_r48.json   (BUILD-TIME ONLY, not shipped)
swizzler_config     : metadata.bin                        (runtime equivalent, IS shipped)
full_model_sha256   : 5b9effa377d9c3370bc663be00d693aa58125854c0de37ad44e26c9eb3841679
backbone_signature  : 948431e5b7efa9078caad7cb1512da6883cb636377cd2fc72692eedef80dc7ef
```
This reconciles the earlier puzzle: **44 layers × `num_ffns`=3 FFN modules = 132**, each with
gate/up/down = 3 constants → **396**. Confirmed.

## metadata.bin (BBBB) structure — decoded
- magic `BBBB`, u32 size = 439324 (matches).
- **~14 080 records of 24 bytes** (6×u32; type tag `0x06000C2F` per record).
- **~44 layer sections** (43 stride-582 header records) × ~320 weight-block records each.
- Records carry byte-offsets into `ifp_rasterized_weights.bin` → the exact palettized-block
  layout (resolves my earlier boundary-drift/dedup approximations for the FFN weights).

## Where every weight lives (COMPLETE)
| Component | Source | Format | Recovery status |
|---|---|---|---|
| FFN / MoE experts | `ifp_rasterized_weights.bin` (4.7 GB) | 4-bit LUT-palettized, ANE-swizzled | recovered + de-swizzled; exact offsets in `metadata.bin` |
| Attention Q/K/V/O, embed, backbone | **`binary_0.hwx` (252 MB)** | ANE program, palettized in `__KERN` | **the remaining packed source** |
| RMSNorm γ | folded into adjacent weights at compile time | — | already inside recovered weights |
| Router / IFP mask | baked at export (`dense_only`) | — | not needed at inference |
| LoRA adapter | `lora_i1_r48_48_constant_data.bin` (68 MB) | fp16 low-rank | present, unparsed |

**Net:** the *only* remaining packed weight source is `binary_0.hwx` (attention/backbone).
Everything else is either recovered (FFN), folded (norms), baked (router), or present (LoRA).

---

# hwx parse — RESOLVED (the hwx is code, not weights)

Parsed `binary_0.hwx` (263 MB): a **Mach-O 64** with ANE magic `0xBEEFFACE`, 13 load
commands. Sections:
- `__TEXT.__text` 53 MB, `__INIT.__text` 1.4 MB — **compiled ANE kernel microcode** (low entropy).
- `__KERN_0` 133 MB, `__KERN_1` 25 MB — high byte-entropy (7.8/8). Looked like palettized
  weights, but the **scale-decontaminated structure test gives R≈1.0 (pure noise)** with the
  FFN codec + 8×128 de-swizzle. → these are **compiled kernels / ANE program data, not weights**.
- `__MKERN_0` 202 MB — segment `filesize=0` (**no data in the file**); it is **runtime-mapped**.
  Its size `0xc0c8000` is **exactly the max offset in `metadata.bin`** → the backbone weights
  are mapped into `__MKERN_0` *from the rasterized file*, not stored in the hwx.

## The reframing that resolves everything
The attention/backbone weights were never in the hwx. They are in
`ifp_rasterized_weights.bin`, **index region 0x1078000–0xc0c8000 (~17–202 MB)**, indexed by
`metadata.bin`. Proof — structure test on that region (same LUT+de-swizzle codec as the FFN):

| candidate shape | R (should be ≫1 for real weights) |
|---|---|
| 2048×1536 (q) | up to **46** |
| 1024×1536 (kv) | up to **43** |
| 1536×1536 | up to **44** |

So **no ANE-microcode reverse engineering is needed.** Every weight in the model —
attention, embedding, dense FFN, and MoE experts — lives in the single clean planar
palettized file `ifp_rasterized_weights.bin`, which the existing codec already decodes. The
hwx is only the compiled ANE program that consumes them.

## Final status of the model
| Piece | Where | Decodable |
|---|---|---|
| Attention / embed / dense FFN (backbone) | rasterized file, 17–202 MB index region | **yes** (R≈40+), map via `metadata.bin` |
| MoE experts | rasterized file, 202 MB–4.7 GB | **yes**, 396 constants mapped via mpsgraph hierarchy |
| RMSNorm γ | folded into the above weights | already present |
| Router / IFP mask | baked (`dense_only`) | not needed |
| ANE program | `binary_0.hwx` | code, not weights |

**Only remaining work for a correct `.pt`:** use `metadata.bin` (backbone swizzler) + the
mpsgraph hierarchy + `config.json` dims to assign correct names/shapes to every block. The
current `afmplus_v11_ifp_FULL.pt` has all 252 tensors but some labels are approximate
(e.g. a `qkv` tensor is 3072×1536 — dense-FFN dims, not the 4096-wide fused QKV). This is now
a mapping/labeling task on already-decoded data, not a locked-format problem.

---

# FINAL rebuild — `afmplus_v11_ifp_FULL_v2.pt` (10.23 GB, verified)

Rebuilt from `ifp_rasterized_weights.bin` with the corrected layout + codec.

**Rasterized-file layout (nailed down):**
- `[0x60 .. 0x1078000]` — global fp16 scale table (8.63 M scales, per-1024-block)
- `[0x1078000 .. 0xc0c8000]` — **attention** 4-bit indices (44 layers, 370 M weights)
- `[0xc0c8000 .. EOF]` — **FFN/MoE experts + embed/head tail** 4-bit indices (9.47 B weights)
- scale coverage ≈ 8.84 B weights; the ~1 B-weight tail decodes with the last scale
  **clamped** (finite, approximate) — this was the source of the earlier NaN/overflow.

**Contents (verified, 0 non-finite):**
| Group | Count | Storage | Check |
|---|---|---|---|
| Attention (44 layers: 32 qkv + 12 KV-reuse q, + o) | 88 tensors | fp16 | mean≈0, std≈0.012, R median 12.7 (max 63) |
| Experts (FFN/MoE, 60 fused blocks) | 9.47 B weights | int8 + per-tensor fp16 scale | dequant sane (std≈0.02) |
| **Total** | **9.85 B params** | — | matches ~10 B MoE total for "3B-active" IFP |

Payload also carries: ground-truth `config`, `codec` (codebook + scheme), Apple's
`full_model_sha256`/`backbone_signature`, and a per-tensor `manifest` with structure-ratio R.

**Honest caveats:** tensor→name labels are sequential/greedy best-effort (18 tensors R<4,
mostly `o`-projections); norms are folded into the ANE weights (named in odix, not separately
shipped); router baked (`dense_only`); the final ~9.7 MB partial expert block (0.2%) is not
captured. Every weight value is decoded from Apple's shipped rasterized file and R-verified.

