# `main-h16g.odix` decompiler — structural map

A partial decompiler for Apple's `odix` container (the FlatBuffer op graph, 289 MB), built to
recover the 396 FFN constants' `(offset, size, shape)` in `ifp_rasterized_weights.bin`. The
reader is `src/odix_fb.py`. This documents how far it gets and the exact remaining gap, so the
next effort starts here rather than from scratch.

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
