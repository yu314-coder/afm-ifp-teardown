# `main-h16g.odix` decompiler — structural map

A partial decompiler for Apple's `odix` container (the FlatBuffer op graph, 289 MB), built to
recover the 396 FFN constants' `(offset, size, shape)` in `ifp_rasterized_weights.bin`. The
reader is `src/odix_fb.py`. This documents how far it gets and the exact remaining gap, so the
next effort starts here rather than from scratch.

## Constant inventory CRACKED via MLIR names (2026-07-15) — the big schema win

The binary type-table resisted, but the **constant NAMES carry the full structure** (MLIR
`_wrapped_model_*` symbols in the string pool). Extracted `odix_constant_inventory.json`:

- **FFN decomposition:** `feed_forward_hidden_transform_linear_0` = **gate**, `linear_1` = **up**,
  `feed_forward_output_transform` = **down**. (Dense layers carry all three as explicit
  `palettized_indices_raw`; sparse FFN is the IFP-fused form.)
- **Attention:** `attention_qkv_transform_wrapped_fused_linear` (standard, full QKV) vs
  `attention_q_transform` (kv-reuse, q-only), `attention_output_transform`, plus
  `attention_qk_norm_{query,key}_norm_weight`.
- **Sandwich norms are EXPLICIT PLAIN constants** (0 of 449 `*_norm_weight` names are palettized):
  `{attention,feed_forward}_residual_connection_{pre_residual,post}_norm_weight` + qk-norms +
  one `output_norm_weight`. NB: the odix is the *export* graph (pre-ANE-compile); the *compiled*
  mpsgraph folds these to parameter-free RMSNorm (see the "norm folding" finding) — both are true,
  and the plain γ DATA lives in the odix constant-data section.
- **193 physical `palettized_indices_raw`** weight constants; **12 dense layers** carry explicit
  gate/up/down.
- **Segments:** `dense_segment` layer 0..11 (12); `sparse_segment_standard_segment` 0..22;
  `sparse_segment_kv_reuse_segment` 0..20 (q-only). The attention partition is standard-vs-kv-reuse;
  the FFN partition is dense-vs-sparse. (Exact reconciliation to `num_layers`=44 still has a
  naming-variant counting ambiguity — segment layer indices may restart per adapter config.)
- Activation tensor-type strings present: `tensor1x{1024,1536,2048,3072}x1x{8,16,64}f16`
  (hidden 1536, dense-FFN 3072, attn 2048/1024; the 8/16/64 = gather/seq batch).

**Per-constant data OFFSETS + the DEPLOYED expert C_out — RECOVERED from `metadata.bin`
(2026-07-15).** The sparse FFN has no per-constant palettized descriptor (it is IFP-fused), so its
layout lives in `metadata.bin` (BBBB container), which we now parse fully:
- **Down-proj address table** (marker `0x0600beef`, 24-byte records, **14,080** of them): each
  record = one **8×1536** 4-bit down tile (6144 B data + 32 B header ⇒ 6176 stride) with its
  physical tile address in col4/col5. 14,080 tiles × 8 rows = **112,640 rows** ⇒
  **3,520 rows ≈ 14 experts per sparse layer** (14,080/32 = 440 tiles/layer).
- **Gate/up address table** (marker `0x0a00b0ef`, **4,223** records, 10,272-B stride).
- **This EQUALS the shipped `ifp1_r48` config: 10 active + 4 shared = 14 experts** ⇒ the metadata
  encodes the DEPLOYED resident expert set, and — crucially — its **physical addresses**, so the
  deployed sparse FFN is *gatherable* (tile addresses known) and, being ungated, summable without a
  router map. Block structure (160-tile ≈ 5-expert / 320-tile ≈ 10-expert groups) mirrors the
  active/shared split.

So the DEPLOYED per-layer C_out (~14 experts) is recovered; only the un-pruned SUPERSET C_out (all
routable experts, variable per layer) remains in the binary symbol pool — and that superset is NOT
needed to run the shipped `ifp1_r48` model. Names give role+layer; `metadata.bin` gives the deployed
shape+address. Deliverables: `odix_constant_inventory.json`.

## Solid (cracked)

- **Root table** (`ir_start + u32(ir_start)`): `field[1]`=12 module ops, `field[2]`=114 values,
  `field[3]`=114 identity index `[0..113]`, `field[9]`=38 functions.
- **`field[9]` = 38 named functions = the export configs**:
  `lora_48_ifp1_r48_{prompt_opt,extend}_{dense_only,sparse_only}_{256..8192}_{8,16,64}`, plus
  `gather_embeddings_{8,16,64}` and `load_embeddings`. Each maps to a consecutive triple in
  `field[2]` (38 × 3 = 114).
- **`field[2]` values**: `f1=[type_index]`, and the type-index sequence is period-3
  `[gate, up, down]` — `[21,2,13]` for the embedding gathers, then `[18,3,18/19]` repeating over
  the layer stack.
- ~~**`NDArray.alloc_const` op**: fields `[flags, offset, size, dtype]` — inline only for special
  constants (e.g. the 393216-byte embedding chunk, dtype `0x40012`).~~ **RETRACTED (2026-07-15) —
  this was an artifact.** All four `u32 == 393216` sites in the IR share the byte pattern
  `00 00 06 00`, i.e. u16 `(0, 6)` where `6` is a FlatBuffer **vtable_size**, always followed by
  `(table_size, 4, 6)`; one sits beside the debug string `line`, another beside `NDArray`/`Scalar`.
  So `393216 = 0x00060000` and `0x40012` are **u32 reads straddling u16 vtable headers**, not a
  size/dtype pair, and no inline `[flags, offset, size, dtype]` record was ever demonstrated.
  (`512 × 1536 / 2 = 393216` is a numerical coincidence with `0x60000` — it made the artifact look
  meaningful.) **Lesson: never u32-scan a FlatBuffer for "sizes"; the vtables are u16.** The 396
  FFN constants carry no inline offset/size; the rasterizer computes offsets from shapes.
- **Expert dims are variable**: IR histogram of multiples-of-256 shows `42–232` experts per
  constant — there is no uniform width (this is why a uniform-width offset model mis-aligns and
  the reconstructed FFN never produced coherent text).
- `metadata.bin` (`BBBB` swizzler) indexes the **attention region only**: its maximum offset is
  `0xC0C8000`, exactly the start of the expert region.

## The remaining gap (bounded schema-recovery)

```
constant → value → type_index (18/3/19…) → TYPE TABLE → dim symbols → SYMBOL POOL → 1536, C_out
                                            ▲ not located          ▲ (only 26 literal 1536s in 32 MB IR)
```

Shapes are **doubly-indirect**: a constant's type is an index; the type table stores dims as
*interned symbol references*; the integers (`1536`, per-constant `C_out`) live in a symbol pool.
Resolving both tables — each in the same shared-vtable / self-relative-`0xff`-ref encoding — is
what remains. It is a self-contained FlatBuffer schema-recovery problem, not open-ended.

Once `C_out` per constant is known: `size = C_out × 1536 / 2` (4-bit) → cumsum from
`BACK = 0xC0C8000` → decode with the validated LUT+de-swizzle (Eq. de-swz in the paper). Coherent
text then still needs three calibration pieces: embedding dequant, output-norm, and layer
pairing.

### Value-table schema (2026-07-15 progress)

The 114-entry value table (`root.field[2]`) is now partially decoded. Each value is a FlatBuffer
table with:

- **`f1` = `type_index`** (a `vec[1]`): values `{1,2,3,13,18,19,21}`; histogram
  `{18:52, 3:35, 19:16, 2:4, 21:3, 13:3, 1:1}`. Types **18 and 3** are the two dominant FFN
  constant kinds; they interleave down the layer stack.
- **`f2` = `dtype` = 12** (a `vec[1]`), constant across all 114.
- **`f0`, `f4`** are **monotone symbol-pool references when scalar** — `f4` runs `357 → 3009`
  and `f0` runs `13 → 286` monotonically across the 114 values (i.e. each value names an entry at
  an increasing offset in a symbol/name pool) — and **serialized shape/data vectors when a vec**.
  Those vecs interleave real dims with `0xff…` self-relative ref-markers, so they do **not** parse
  as clean int arrays.

**Expert widths ARE present** in the `f0`/`f4` vecs (as multiples of 256): observed
`232,140,126,65,57,32,20,17,15,13,12,4,3,2,1` experts — spanning the full "42–232 per constant,
variable width" range and **confirming per-constant variable expert width** (the root cause of the
uniform-stride FFN mis-alignment). But a *clean* per-constant `C_out` still needs the type table
resolved: `type_index` reaches 21, yet no root vector has ≥22 entries (`field[0..9]` sizes are
`6,12,114,114,-,-,1,-,-,38`), so the type table is **nested inside a module op or a value's own
sub-tables**, not a top-level vector. Reader: `src/odix_fb.py`. This remains the self-contained
(but genuinely hard) schema-recovery gap.

**Caveat added 2026-07-15:** even a complete `C_out` map does **not** yield coherent text — the
token embedding is now proven ANE-baked and absent from all shipped files (see the teardown paper
§embedding / `find:embed`), so a from-weights standalone forward is not reachable regardless.

### Schema crack (2026-07-15) — module ops + debug tree decoded; type→shape core still binary

Advanced the decode; the doubly-indirect `type_index → shape` core remains the hard part.

- **`alloc_const` op descriptor cracked.** Each of the 5 global `NDArray.alloc_const` ops
  (`field[1]` op[7..11]) carries `f0 = [flags, offset, size, dtype]` (op[11] = `[1, 0, 393216,
  0x40012]` = the [512,1536] embedding chunk) and a generic `f1 = VEC[19]` whose 5-word prefix
  decodes to the ASCII op name `"NDArray.alloc_const."`. The `VEC[19]` tail has self-relative
  `0xff…` REFs (into the symbol pool) plus per-op scalars (op[11]:1408/209/340, op[8]:232/128/331)
  that do **not** cleanly decode to dims — so shapes are not in the op descriptor beyond `size`.
- **The IR debug/location tree is readable** (keys `location_type, name, op_id, named_child_loc,
  sub_locations, line, filename, column`): it yields op NAMES + hierarchy + `op_id`
  (`TransformerAttention_274`, `PalettizedConv2D`, `LoRAFusedMultiOutputLinear_745`,
  `ANE_FusedMultiOutputLinear_670`, `mul_7025`) and MLIR attrs (`{id = N : ui64, level = "coreai"}`).
  Gives names/op_ids, **NOT shapes**.
- **Value-table semantics** (`field[2]`, 114 entries): `f1 = type_index` ∈ {1,2,3,13,18,19,21},
  `f2 = dtype = 12`, `f0/f4` = serialized shape/data vecs (expert widths 232,140,65,57,32,20,13,12
  × 256 are present, but interleaved with `0xff…` refs so they don't parse as clean int arrays).
- **STILL UNCRACKED:** the `type_index → type-table → interned dim-symbol → symbol-pool → {1536,
  C_out}` chain. The type table is not a top-level vector (indices reach 21; a BFS over reachable
  tables found no clean 22-entry candidate — it explodes into data). The REFs from constants point
  to MLIR-attribute / debug strings, not clean dim arrays. Extracting clean per-constant `C_out`
  needs following op_id→data references through the FlatBuffer's own (unknown) schema — the genuine
  multi-week task. **Value now reduced:** even a full `C_out` map won't produce text (embedding
  wall), so this is worth doing only to firm up the FFN structural record, not to reach generation.
