# PICO byte-exact weight ARRANGEMENT — HONEST VERDICT (2026-07-18)

Scope: synthesize the five arrangement analyses (`__INIT` descriptors, `__TEXT` op-stream,
intra-block order, K/V shape, coreml2hwx route) into a single honest verdict on whether the
byte-exact composition of pico's 998 weight blocks into `[Cout,Cin]` matrices is now PROVEN.

Source asset (verified on disk, contrary to one analysis's local-dir-only search):
`…/purpose_auto/031c7be6…asset/.AssetData/model.odixpackage/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx`
(193,921,024 B, magic `0xbeefface`, H16 ISA v17). No Apple weights committed.

---

## BOTTOM LINE — **NOT PROVEN** (the ANE-op-stream / hardware-convention floor holds)

The byte-exact arrangement — a written-out element→position bijection for the 998 blocks — is **NOT
proven**. Nothing in the shipped program spells it out; it reduces to a **fixed ANE conv weight-layout
convention** that (i) is not stated as data in pico's `binary_0.hwx`, (ii) is provably **invisible to any
singular-value / structure test**, and (iii) can only be seeded from `coreml2hwx`, whose container does
**not** match pico's tile geometry. This is exactly the wall `PICO_WEIGHT_RESULT.md` and
`PICO_H16_ISA_SCOPE.md` predicted.

That said, this is **not "nothing."** Three structural facts are now **PROVEN** and independently
re-verified in this pass, which meaningfully constrain the unknown. So the accurate label is
**PARTIALLY constrained, byte-exact arrangement NOT proven** — with a clean split between what is pinned
and what remains open on each of the three sub-questions (tile-order / block-order / K-V-shape).

---

## PROVEN (independently re-verified against the binary this pass)

1. **Block CONSUMPTION SEQUENCE = symbol order = program order = `pico_weight_map.json` order.**
   The kern0-relative block bases `0x0, 0x20800, 0x41000, 0x61800, 0x82000, 0xa2800, 0xc3100, 0xe3900,
   0x104100, 0x124900, …` are real `LC_SYMTAB K<hash>_ne_0` bases, appear as u32 in `__TEXT` (18–48× each,
   across specialization variants), are **absent from `__INIT`**, and ascend in exactly the map's
   Q0,Q1,Q2,Q3,K,V,O0,O1,O2,O3,gate… order. The map's block *sequence* is therefore correct and
   op-stream-confirmed.

2. **No per-tensor tile-permutation table exists — the arrangement is a FIXED convention, not shipped
   data.** Across `__TEXT` region 0, 1107/1107 conv ops carry a strictly monotonic `k·stride` tile table;
   0 are permuted. The 16 tiles are DMA-fed `ne_0…ne_15` in canonical order. This **rules out a hidden
   per-tensor scramble table** — a real elimination.

3. **Every N-block is PHYSICALLY a 512×512 grid (4×4 of 128×128 tiles); a physical `[1024,256]`/
   `[1024,128]` tiling is refuted.** From the symtab (16 `_ne` tiles/block at `0x2080`) plus the op stream
   (one conv per 512×512 block). The `__INIT` dim-field census reproduces exactly: the pair-encoded field
   `(0xff000000, dim−1)` is **512-only** — 1728× for `0x1ff`(512), **0×** for `0xff`(256), `0x3ff`(1024),
   `0x7f`(128), `0xc7f`(3200). There is no 256/1024/3200 op-tile dimension anywhere.

---

## NOT PROVEN — the actual byte-exact arrangement (the three sub-questions)

### (a) tile→grid-cell mapping + intra-128×128 order — **OPEN (hardware convention)**
Which of the 16 canonically-fed `ne` tiles lands in which 4×4 cell, and the byte order inside each
128×128 tile, are **not** encoded per-op. They are the ANE conv weight-layout convention. Two independent
walls confirm this cannot be closed from what's shipped here:
- **SV-invisible (proven, re-derived):** tile→grid reassignment and intra-tile row/col order are
  permutation (+ diagonal-scale binding) operations, under which singular values are *exactly* invariant;
  `picolib.smax` "R responds to arrangement" is a 4-iteration **non-convergence artifact**, not structure.
  No R / structure test can rank a correct de-swizzle above a wrong one. (This is why the earlier "R≈3.8,
  not clean" observation is chasing a metric that is blind to the target.)
- **coreml2hwx gives a formula for the WRONG container:** the route runs and yields a bijection-validated
  closed form, but for a **64×1024 strip** tile (64 B header, 65536 nibbles/block) — pico's real tile is a
  **128×128 square** (128 B header, 16384 nibbles). Mismatched on every geometric axis and invariant to all
  reachable levers (S, conv/inner_product, spatial, OCG). Direct transfer is invalid — same failure mode as
  the 3B (`coreml2hwx 16×768 ≠ Apple 48×256`).

### (b) block→quadrant placement within `[Cout,Cin]` — **OPEN**
Block *sequence* is proven (above), but **which** block occupies `[0:512,0:512]` (top-left) vs the other
quadrants of a 2×2 tensor (Q/O) is **not** decodable: region-0 weight-conv records contain **zero** `__DATA`
output pointers (all 219 activation pointers live in the separate activation ops), and the per-op `+80`
field is compiler scratch, not a channel index. Output placement is bound indirectly via dsid relocation.
"Sequence order == raster (TL,TR,BL,BR)" remains an **assumption**, not a proof.

### (c) K/V matrix shape/semantics — **PARTIALLY resolved; logical shape OPEN**
- **Solid:** the K and V blocks are each a single **512×512** physical grid (as in every N-block).
- **Overclaim corrected:** the analysis asserting "**[512,512] PROVEN**, `[1024,256]` refuted" is
  **overstated**. Its two pillars fail scrutiny: (1) "no 256 dim in `__INIT`" — but `__INIT` was
  independently confirmed to hold **0/998 weight addresses** (kern base `0x394d8000`: 0×), so it is the
  **activation/DMA schedule** and its 512-unit tiling describes *activations*, not the *logical weight
  matrix* shape; (2) the "4×4 wins 21/24" rank test is exactly the SV-invisible metric debunked in (a).
  Decisively, the analysis that actually read the **weight-conv operands** lists `KV_true_shape` as
  **UNRESOLVED** (Cin=1024 declared; the `+71/+74 = 0x20/1` fields don't cleanly factor to
  `[1024,256]` vs `[1024,128]` vs `[512,256]`) — a direct contradiction resolved in favor of "open."
- **Net:** the *physical block geometry* is 512×512; the *logical* K/V shape and the fused-KV-`[512,1024]`-
  contraction-halves reading are **inferred, not binding** — the op→weight binding is dsid-relocated, at the
  information wall.

---

## Cross-analysis contradictions, resolved

- **"[512,512] PROVEN" (K/V analysis) vs "KV_true_shape UNRESOLVED" (op-stream analysis):** resolved as
  UNRESOLVED. The op-stream analysis read the weight operands directly; the K/V analysis leaned on `__INIT`
  (proven weight-address-free) and a debunked rank test.
- **"R responds to arrangement" (implicit in every structure-test claim) vs "R is SV-invisible":** resolved
  as SV-invisible — verified that `smax` is non-convergent and true SVD is permutation-invariant. All
  `assembled_R` / `best_shape` / "4×4 vs 8×2" signals are artifacts and must not be used as arrangement
  evidence.

---

## Genuine correctness finding (orthogonal to arrangement — VERIFIED this pass)

The 128 B N-tile header's first 32 bytes are a **per-tile 16-entry non-uniform fp16 codebook (LUT)**, not a
scalar scale. Verified on Q0/K/V of layer 0: monotone, NF4/Lloyd-Max shape (edge step ~0.18–0.24, center
~0.06–0.07, **edge/center ratio ~2.5–3×**), distinct per tile; a uniform `(nibble−7.5)·S` fit leaves
**14–19 % max residual**. The correct dequant is `W = codebook_tile[nibble]`. This means `picolib`'s linear
decode is a **value approximation** (the N-blocks are LUT-palettized like the L/down blocks, just with a
16-entry codebook), and `PICO_WEIGHT_RESULT.md`'s "genuine affine-int4" characterization is approximate.
**It does not affect arrangement** (a monotone per-tile value remap), so the map/geometry conclusions stand;
it is worth folding into the decode recipe for weight accuracy.

---

## Map / result files — deliberately NOT rewritten with an arrangement

Because the byte-exact arrangement is **not** proven, `pico_weight_map.json` and `PICO_WEIGHT_RESULT.md`
were **not** given a (would-be false) arrangement recipe. Their existing honest statement —
"168/168 located; byte-exact `[Cout,Cin]` composition 0/168 proven" — remains correct. The only
justified additions (not made here to avoid overclaiming without a second reviewer) would be: (i) note that
block *sequence* = program order is now op-stream-confirmed; (ii) flag the per-tile codebook-LUT value fix.

---

## Independent verification log (this pass, on the real binary)

| check | result |
|---|---|
| segment table | `__INIT` foff 0x2f8000 / vm 0x3750c000; `__TEXT` foff 0x3d4000; `__KERN_0/1/2` @ 0x22c4000/0xa2c0000/0xb35c000 — matches `picolib.SEG` |
| block count | 998 (kern0 854 / kern1 108 / kern2 36) |
| `__INIT` dim pair `(0xff000000,X)` | 512→1728, 256/1024/128/3200→**0** |
| `__INIT` weight addresses | abs vaddrs 0/200, kern base `0x394d8000` **0×** → no weight addressing |
| `__TEXT` weight offsets | Q/K/V/O kern-relative bases present 18–48× each; absent from `__INIT` |
| block order | symtab bases strictly ascending == map order == op-stream order |
| tile-perm table | not present (op stream feeds ne_0…15 monotone) |
| per-tile header | 16-entry monotone non-uniform fp16 codebook; linear fit resid 14–19 % |

**Verdict: NOT PROVEN (byte-exact) — partially constrained. Pinned: block sequence, absence of any shipped
permutation table, 512×512 physical block geometry. Open: intra-block tile→grid + intra-128 order (a),
block→quadrant placement (b), logical K/V shape (c). The remaining gap is a fixed ANE hardware convention,
SV-invisible and not present as data in the shipped program — the op-stream/ISA-operand floor holds.**
