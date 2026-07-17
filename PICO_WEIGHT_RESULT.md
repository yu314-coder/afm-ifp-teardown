# PICO (H16, on-device ~300M) weight recovery — FINAL RESULT

Date: 2026-07-18
Map file: `/Volumes/D/fix/pico_shapes/pico_weight_map.json` (169 entries: 168 logical tensors + 1 flagged partial unit)
Decode library: `/Volumes/D/fix/pico_shapes/picolib.py`
Source asset: `binary_0.hwx` (H16 ISA v17 ANE program), symbol-table–driven; **no Apple weights committed**.

---

## Bottom line

**MOSTLY — 168/168 logical tensors are LOCATED and their blocks are decode-validated as genuine
affine-int4 weight data, with an exact reproducible decode recipe; but the byte-exact composition of
those blocks into `[Cout,Cin]` matrices is NOT independently proven, and the 24 `down` projections use
a different (palettized) codec that the uniform path cannot score.** The weights are found and the recipe
is known; what is *not* pinned down is intra-tensor geometry (block order, tile order, the FFN packing,
and even the K/V matrix shape). Per the scope docs this last step is an information wall — it needs the
ANE op-stream operand bindings, not a better structure test.

---

## What the packing is (validated)

The 998 weight blocks in `__kern_0/1/2` are named in `LC_SYMTAB` as `K<64-hex>_ne_<0..15>` — **16 tiles
per block at the `0x2080` stride**. They fall into **three physical block classes**, distinguished by the
intra-block tile stride read from the `_ne_k` vaddrs (not from any label):

| class | count | tile stride | int4 / block | meaning |
|---|---|---|---|---|
| **N** | 848 | `0x2080` | 262,144 (512×512) | uniform int4 |
| **s** | 50 | `0x1080` | 131,072 (1024×128) | the 3200-dim column remainder half-block |
| **L** | 100 | `0x6480` | 819,200 | 16-entry fp16 codebook (palettized) — the `down` projections |

The class sequence in program order is perfectly periodic:
**`[N×10][s][N×12][s][N×12][L×4]` = 40 blocks/layer**, and the int4 budget closes exactly:
2,621,440 (attn) + 6,553,600 (gate+up) + 3,276,800 (down) = **12,451,840 int4 / layer**.

- **24 complete 40-block layers** (L0–L23) → exactly **24 × 7 = 168 logical tensors** (960 blocks).
- **1 trailing partial unit** (38 blocks, 2 N-blocks short of a full layer; lives across
  `__kern_1`/`__kern_2`). Its s/N interleaving is non-canonical, so it is **not** force-split into
  Q/K/V/O — recorded as a single flagged `PARTIAL_UNIT` entry. The 2-block shortfall is most likely
  content-hash dedup (symbol names are SHA-style content hashes, so bit-identical blocks collapse) or a
  genuinely reduced final layer.
- 960 + 38 = **998 blocks, zero duplication, full coverage.**

---

## Decode recipe (exact, reproducible)

For any block at symbol-table vaddr `va` in segment `ns ∈ {8,9,10}`:

1. **Symtab → file offset.** `SEG = {8:(0x394d8000,0x22c4000), 9:(0x414d4000,0xa2c0000),
   10:(0x42570000,0xb35c000)}`; `foff = va − vmaddr_base + file_base`.
   (Segments `__kern_0`=8, `__kern_1`=9, `__kern_2`=10.)
2. **16 tiles per N-block, stride `0x2080`.** Each tile = 128-byte header (per-tile fp16 scales; decodes
   as a smooth ramp) + `128×128` int4 packed 2-per-byte (8192 B).
3. **Nibbles → signed:** `low = byte & 0xF`, `high = byte >> 4`, value `= (nibble − 7.5)`. Reshape
   `128×128` and **transpose** (`.T`).
4. **Arrange** the 16 tiles into the block grid (`arrange(T,gr,gc)` = `np.block`), then compose the
   constituent blocks into the logical `[Cout,Cin]` matrix.
5. **Scale:** multiply by the per-tile fp16 scale from the 128-byte header → real weights `≈ (q−7.5)·S`.
6. **Structure test** `R`: scale-decontaminated top-singular-value ratio vs. a shuffled baseline; real
   int4 weight ≈3.6–6, scrambled ≈1.4.

`down` (class **L**) does **not** use this uniform path — it is 16-entry codebook/grouped-palettized
storage (the separately-cracked ANE down-proj codec). The uniform `(nibble−7.5)` read of an L-block is
meaningless, so its `R` is invalid by construction and it is validated structurally only (stride +
codebook header + exact size + count).

---

## Per-role structure results (across the 24 complete layers)

`mean_R` = mean per-block `best_shape` R; `assembled_R` = R of the composed `[Cout,Cin]` matrix.
Both were **independently reproduced** (not read from the JSON) by two adversarial validators covering
L0–11 and L12–23.

| role | shape | n/24 | blocks/tensor | mean_R range | assembled_R range |
|---|---|---|---|---|---|
| Q | 1024×1024 | 24 | 4 N | 4.07–4.96 | 7.43–8.84 |
| K | 1024×256 | 24 | 1 N | 2.90–4.87 | 2.02–4.77 |
| V | 1024×256 | 24 | 1 N | 3.56–5.54 | 3.22–5.54 |
| O | 1024×1024 | 24 | 4 N | 3.57–4.53 | 6.80–8.89 |
| gate | 1024×3200 | 24 | 1 s + 12 N | 4.02–4.45 | 10.95–12.13 |
| up | 1024×3200 | 24 | 1 s + 12 N | 3.59–4.13 | 8.87–11.05 |
| down | 3200×1024 | 24 | 4 L | 1.44–2.79* | n/a (palettized) |

\* `down` mean_R is the uniform-int4 read of a palettized block — expected noise-floor, not a defect.

---

## What is validated vs. what is not

### Validated (load-bearing, independently reproduced)
- **Block identification is real and exact.** All block offsets across all 24 layers resolve to genuine
  `K<hash>_ne_0` symbol-table tensor bases (0 misses), consecutive in program order with the correct
  per-class strides.
- **Every N/s block is genuine affine-int4 weight data** — per-block R elevated well above the 1.4
  scramble floor; clean bell-shaped int4 histograms peaked at 0.
- **Block counts and the int4 budget close exactly** for all 168 tensors (`int4_ok` true everywhere):
  Q/O=4 N, K/V=1 N, gate/up=1 s+12 N (correctly accounting for 3200 not dividing 512), down=4 L.
- **The per-layer physical byte-size signature** `[10×N | s,12×N | s,12×N | 4×L]`, computed **only** from
  file-offset gaps (ignoring the JSON labels), matches all 24 layers. The two half-size `s` blocks land
  exactly at `gate[0]`/`up[0]`; the four L blocks land exactly at `down`.
- **`down` correctly flagged** as a different (palettized) codec; `assembled_R: null` is honest.
- **Allocator artifact resolved (L21):** `up`'s `s` block was slotted into leftover `__kern_0` space and
  looks non-contiguous, but the grouping is *forced* (exactly two half-blocks in that region ⇒ they must
  be `gate[0]`/`up[0]`). Correct, not a bug.

### NOT validated (do not trust as composition evidence)
- **`assembled_R` is a concatenation artifact, not a grouping proof.** Negative controls are decisive:
  cross-role, cross-layer, reversed, and shifted block tuples all produce R≈7–11 indistinguishable from
  the "correct" assembly. Stacking any N genuine int4 blocks inflates the top singular value (block-scale
  DC structure). **`assembled_R` should be disregarded as a correctness signal** — it confirms the blocks
  are real, not that they form that particular projection.
- **Intra-tensor geometry is unresolved:** `best_shape` disagrees per layer on the sub-block reshape
  (512×512 vs 256×1024 vs 128×2048), some tensors need a non-row-major 2×2 block order, and the FFN
  8×25 tile packing is not pinned. The Q/O 2×2 placement, intra-symbol tile order, and FFN layout are
  unproven.
- **The K/V shape `[1024,256]` is not structurally supported.** For K/V the geometric 8×2 (1024×256)
  arrangement is *never* the best; 4×4 (512×512) or 256×1024 always wins, and several K/V collapse near
  the 1.4 floor under a rigid 8×2 read. The block is genuine weight data, but `[Cout,Cin]=[1024,256]` is
  an architecture assumption the metric cannot confirm and sometimes contradicts (an intra-tile z-order
  nuance identical in both halves).
- **K-vs-V assignment is by export convention, not proof.** Within the 10-block attention group the
  `{4,1,1,4}` split is size-forced, but which single N-block is K vs V follows the standard Q,K,V,O
  export order — R cannot distinguish two equally-valid weight tiles.
- **The trailing partial unit's exact role layout** is unresolved (flagged, not force-split).

---

## Honest characterization of "complete"

- **Located + identified as genuine weight blocks:** 168/168.
- **Uniform-int4 decode recipe exact & reproducible:** 144/168 (all but `down`).
- **`down` (24 tensors):** located + structurally validated; needs the separately-cracked palettized
  codec to yield actual weights.
- **Byte-exact `[Cout,Cin]` matrix composition rigorously proven:** 0/168 — the structure test cannot
  prove block/tile order, and `assembled_R` is artifactual.

Nothing is mis-located; every block is genuine int4. But the geometry (block order, tile order, FFN
packing, K/V shape) is unconfirmed — exactly the wall the scope docs predicted: pinning per-tensor
arrangement requires the ANE op-stream operand bindings (task-1/3 ISA framing + graph decode), not a
structure test. That is the true remaining floor.

---

## Scripts / provenance
- Map: `pico_weight_map.json`; decode lib: `picolib.py` (`TENS`, `best_shape`, `tiles`, `arrange`, `R`).
- Adversarial validators (scratchpad): `val.py`–`val5.py` (L0–11, incl. negative controls + K/V sweep);
  `validate.py` (L12–23, byte-size signature + intra-tile-shuffle floor control R=1.09).
