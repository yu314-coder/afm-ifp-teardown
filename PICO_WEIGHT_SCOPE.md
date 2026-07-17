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

- **Container.** `binary_0.hwx` is the ANE `hwx` format (magic `0xbeefface`, header
  `fields=128,7,2`), 193.9 MB. First ~32 MB = ANE program + tables + LoRA-specialization sections
  (`__lora_*`, `__arg0/1/5`); the weight-bearing region is **`~0x0228_0000` → EOF (~161 MB)**.
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

Status: **format + region + structure done; tiling + boundaries + scales + assembly remain.**
