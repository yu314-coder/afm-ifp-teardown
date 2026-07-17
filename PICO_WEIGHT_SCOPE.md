# Pico Weight Teardown — Scope of Work (evidence-based)

Scoping the recovery of the **transformer weights** of the draft model
`afmplus-v11.0-pico` (the 300M model characterized in the paper, §`sec:pico`). Its embedding is
already recovered dynamically (§`find:embrecover`); this document is about the attention + FFN
weights, which are **ANE-baked** in `binary_0.hwx` (pico is dense, so — unlike the 3B — there is
**no separate `ifp_rasterized_weights.bin`**; everything is in the hwx).

Everything below is from static inspection of the operator's own on-device asset. No Apple weights
are or will be committed — only structure, method, and validation statistics.

---

## 0. What is already cracked (this pass)

- **Container + segment map.** `binary_0.hwx` is a Mach-O-like ANE image (magic `0xbeefface`,
  cputype `0x80`=ANE, `ncmds=13`), 193.9 MB. Parsed `LC_SEGMENT_64` load commands give the exact
  boundaries:
  | segment | fileoff | filesize | role |
  |---|---|---|---|
  | `__TEXT` | `0x003d4000` | 32.4 MB | ANE program code |
  | **`__KERN_0`** | **`0x022c4000`** | **134.2 MB** | **main weight blob** |
  | `__KERN_1` | `0x0a2c0000` | 17.4 MB | weights |
  | `__KERN_2` | `0x0b35c000` | 5.8 MB | weights |
  | `__MKERN_0`,`__MKERN_9` | — | 0 (89 MB vmsize) | LoRA mutable kernels, runtime-patched |
  `__KERN_{0,1,2}` total 157 MB ≈ 315 M int4 params ≈ pico's ~299 M transformer weights. The
  structured tensor found below (`0x46fa400`) sits inside `__KERN_0`.
- **Quant format = affine int4, NOT palettized.** The int4 nibble histogram over the weight region
  is **bell-shaped, centered at 7–8** (`[2117, 2992, 5421, …, 14176, …, 2531, 1099]`), the signature
  of `value = (q − Z)·S` with **Z ≈ 7.5** and Gaussian weights — *not* the 3B's peaked
  shared-codebook palettization. This is **simpler** than the 3B codec (no LUT to recover). fp16/int8
  views are garbage, confirming packed 4-bit.
- **Scales located.** fp16 scale-table regions sit just before/within the weights
  (`~0x01c8_0000`, `~0x01f0_0000`, `~0x0218_0000`; fp16-in-(1e-4,1) fraction ≈ 0.94), consistent with
  per-channel/per-block scales `S`.
- **Real weight structure CONFIRMED + metric recalibrated.** A decode (affine int4, Z=7.5,
  contiguous `[1024,1024]`) at `0x46fa400` scores **R ≈ 5.0** (scrambled ≈ 1.4). Crucially, **R≈5 is
  the *clean* ceiling for affine int4**, not a partial result: a synthetic real low-rank weight
  quantized to affine int4 tops out at R≈3.6, and pico scores *above* that. The "R = 8–11" target
  from the 3B applies only to its *palettized* weights; affine-int4 weights structure-test lower by
  construction (16 levels, per-channel scale stripped). So this tensor is **cleanly decoded**, and no
  ANE de-swizzle is needed for it — a systematic tile search (contiguous, 8×128, 128×8, 32×32,
  16×64, nested `p(o,i)`, both grid/tile orders) leaves **contiguous strictly best**, i.e. pico stores
  weights ~row-major, unlike the tiled 3B.
- **...but the region is not uniformly contiguous.** Sweeping `[256,512]` contiguous probes across
  ~4 layers, only ~3 % structure-test (>3.5), at a **~6 MB (≈ one-layer) period**. So individual
  tensors decode, but most offsets do not sub-block-probe as clean weights — the large FFN tensors
  (`1024×3200`, `3200×1024`) and exact tensor starts are not resolved by blind probing. Per-tensor
  boundaries (kernel-symbol table) are the gating unknown, not the codec.
- **Architecture is fully known** (§`sec:pico`): 24 dense layers × {Q 1024×1024, K/V 1024×256,
  O 1024×1024, gate/up 1024×3200, down 3200×1024} ⇒ **~168 weight tensors** to locate + decode.

- **Layout is 64 KB-tiled (from the ANE program).** `__TEXT` contains **3053 distinct `__KERN_0`
  vmaddr references** (weight-load operands); their dominant gap is **65536 bytes**, so the ANE
  addresses weights in **64 KB tiles** (131072 int4 each). A `[1024,1024]` tensor = 8 tiles, a
  `[1024,256]` = 2, an FFN `[1024,3200]` = 25. The contiguous R=5 is because tiles are ~row-major
  internally; the residual gap to a fully clean decode is the **intra-64 KB-tile order** (a bounded
  de-swizzle over 131072 elements) plus the tile→tensor assignment. The manifest/mpsgraph carry **no**
  per-tensor offset table (only the file name), so the tile→tensor map must come from decoding the
  `__TEXT` ANE weight-load ops (the 3053 refs, ordered by the program) or a structure-test walk.

## 1. What remains (the actual work)

1. **Per-tensor boundaries — now the gating unknown** (tiling is *not* the crux for pico: contiguous
   is strictly best, so no `coreml2hwx` de-swizzle hunt is needed). The hwx section table is not
   simple u32 (offset,size) pairs; the 168 tensor offsets must come from the ANE program's
   kernel-symbol table (the 3B parse in `afm_odix/hwx_expert_dma.json` exposes `kernel_symbol_starts`
   / `segments` — reuse that parser) or a per-shape structure-test base sweep. With exact starts, each
   `[Cout,Cin]` tensor decodes as contiguous affine int4 `(q−7.5)·S`.
2. **Confirm the big FFN tensors.** The `[256,512]` sweep found clean spots only ~once per 6 MB
   (≈ one layer); the `1024×3200` / `3200×1024` FFN tensors need to be probed at their true shape
   (a sub-block of a wide matrix under-reads its structure), not as `[1024,1024]`.
3. **Scale pairing.** Match each int4 tensor to its fp16 scale block (per-channel vs per-1024-block).
4. **Validation = structure test only.** As with the 3B, **every per-layer activation is
   ANE-internal** (paper §`find:aneint`), so there is **no forward-level ground truth** for pico
   either. Each tensor can be validated by the low-rank structure test (R ≈ 8–11 target) and by
   whole-model residual-stability, but an end-to-end greedy-token match is **not** achievable from
   the shipped assets — the same information limit documented for the 3B.

## 2. Effort and odds (honest)

- **Tractable, but a grind.** The format is cracked and structure is confirmed, so this is not a
  wall like the 3B *embedding* (which is genuinely ANE-locked). It is a bounded RE effort:
  coreml2hwx tiling ground truth (proven method) + kernel-symbol boundary parse + 168-tensor decode,
  each validated by the structure test. Estimate: **days–weeks** of focused work.
- **The ceiling.** Even fully decoded, a *from-weights standalone* pico that emits Apple's exact
  greedy token is **not** verifiable per-layer (activations ANE-internal). The deliverable would be
  a structure-validated, residual-stable weight set — the same status the 3B linear weights reached —
  plus the (dynamically-harvestable) embedding. Coherent generation would still ride on the same
  no-ground-truth alignment limit.

## 3. Recommended order

1. `coreml2hwx` a `[1024,1024]` probe → recover pico's exact tile permutation; confirm R→8–11 on the
   `0x46fa400` tensor. (Single decisive experiment; go/no-go on the tiling.)
2. Parse `kernel_symbol_starts` from the hwx to get tensor offsets; pair scales.
3. Decode all 168 tensors; structure-test each; assemble the 24-layer forward; check residual
   stability (not per-layer correctness).
4. Harvest the embedding subset needed for any demonstration (§`find:embrecover`).

Status: **DECODE effectively solved; boundary-enumeration + scales + assembly remain (mechanical).**

### Decode conclusion (strong evidence)
The pico weight decode is **contiguous affine int4** `(q − 7.5)·S` — *no ANE de-swizzle*, unlike the 3B:
- R = 4.99 contiguous; every tile de-swizzle tried (8×128, 16×128, 32×128, nested `p(o,i)`, both
  grid/tile orders, intra-64 KB-tile variants) is **≤ 5.09**, i.e. none beats contiguous.
- Metric calibration: a synthetic real weight quantized to affine int4 ceilings at R ≈ 3.6; pico
  scores *above* that (4.99), so R≈5 is the clean-decode signal for this quantization, not a partial.
- 64 KB tiles hold 128×1024 int4 stacked row-major over output channels ⇒ the tile arrangement is
  already contiguous, and the intra-tile order is ~row-major.

So a per-tensor decode is: read `[Cout,Cin]` int4 contiguously from its `__KERN` offset, `(q−7.5)·S`.
The only open mechanical items: (1) the 168 tensor **offsets** (order the 3053 `__TEXT` `__KERN_0`
refs by program sequence, or structure-walk at each known shape), (2) **scale** pairing (fp16 blocks
at `~0x01c8_0000`/`0x01f0_0000`/`0x0218_0000`), (3) decode + structure-validate + residual-stability
assemble. No remaining information wall on the weights themselves — only the (ANE-internal-activation)
limit on end-to-end token verification, identical to the 3B.
