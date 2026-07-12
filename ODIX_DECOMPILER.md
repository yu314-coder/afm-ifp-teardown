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
- **`NDArray.alloc_const` op**: fields `[flags, offset, size, dtype]` — but **inline only for
  special constants** (e.g. the 393216-byte embedding chunk, dtype `0x40012`). The 396 FFN
  constants do **not** carry offset/size inline; the rasterizer computes offsets from shapes.
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
