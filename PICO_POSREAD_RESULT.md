# Positional read of pico's down-projection z-order

Run 2026-07-19, after [PICO_FFN_ALIGNMENT_RESULT.md](PICO_FFN_ALIGNMENT_RESULT.md) §8 established that
weight-statistics alignment tests are vacuous on AFM and ANE ground truth is the only remaining route.

## 1. pico's real down-proj ANE configuration

Parsed from pico's own `binary_0.hwx` (193,921,024 B, `/System/Library/AssetsV2/…/purpose_auto/031c7be6…`),
ANE Task 218:

```
InDim  : W=64 H=1 C=3200        OutDim : W=64 H=1 C=256
MacCfg : TaskType=0  ActiveNE=4  SmSrc=1  OutTrans=1  FillLowerNE=0
NECfg  : OCGSize=4 FatTileEn=0 WUStack=0
KernelCfg: Fmt=FLOAT16 Pal=1(4bit) SparseEn=0 Reuse=0 SBS=0 Asym=0
16 x CoeffBase/CoeffSize, stride 0x6480
```

Three structural facts follow, none of which were previously known:

- **down `[3200 → 1024]` is 4 ANE tasks of `Cout=256`.** These are exactly the 4 "blocks" in the
  weight map — they are per-task coefficient allocations, not a spatial tiling.
- **The 16 "tiles" per block are the ANE's 16 coefficient banks**, and banks partition *output
  channels*: 256/16 = 16 outputs per bank. Each bank therefore holds **16 outputs × all 3200 inputs**
  = 51200 nibbles, matching the observed 25600-byte payload exactly.
- **`OutTrans=1`.** The 3B tile whose z-order was cracked bit-exactly is `OutTrans=0` (its own
  metadata records this). That is a different output transform, which explains directly why
  transplanting the 3B formula onto pico produced noise.

A census over the file gives 72 such tasks = 18 layers × 4, alongside 18 `C=3200→C=32` tasks (one per
layer), consistent with 4 down-proj tasks per layer in this hwx segment.

## 2. Reproducing the configuration

`coreml2hwx` on a 1×1 4-bit palettized conv `Cin=3200 → Cout=256, S=64` reproduces the geometry exactly:

```
                     pico          probe build
InDim / OutDim   C=3200 / C=256   C=3200 / C=256   match
OCGSize                4               4           match
ActiveNE               4               4           match
Pal                 1(4bit)        1(4bit)         match
CoeffSize[0]        0x6480          0x6440         64-byte delta
OutTrans               1               0           MISMATCH
banks                 16              16           match
```

Emitting the full 1024-wide output in a single task instead yields `OCGSize=5` and
`CoeffSize=0x19040`, confirming that pico's 4×256 split is what produces OCG=4.

`OutTrans=1` could not be triggered by graph shape — bare conv, `mul→conv` (pico's real SwiGLU
fragment, since the down-proj is fed by an EW Mul at C=3200), `conv→add`, and `mul→conv→add` all
compile to `OutTrans=0`. The 64-byte `CoeffSize` delta is a **scale table**: pico's 128-byte header is
codebook(32) + zeros(32) + **16 fp16 scales(32)** + 32 unknown, whereas the probe emits a 64-byte
header with no scales, and does so for every weight distribution tried (integer-uniform,
per-channel-scaled, gaussian) — so it is a property of the compiled mode, not of the data.

## 3. The positional read

Seven probes (`o0, o4, i0, i4, i8`, plus `all0`/`allF`), each with the 4-bit index at (o,i) encoding a
base-16 digit. Two practical corrections were needed:

- **`all0`/`allF` cannot isolate weight bytes here.** A constant tensor has one distinct value, so the
  LUT collapses to a single entry and every index is 0. Payload positions were instead taken
  structurally (16 banks × [64-byte header + 25600-byte payload]), which reproduces the expected
  819200 nibbles exactly.
- **Digits must be decoded through each probe's own codebook.** `o0/o4/i0/i4` compile to an identity
  LUT `[0,1,…,15]`, but `i8` (which spans only 0..12, since 3200 < 16³) gets
  `[0,1,2,2,3,4,5,6,6,7,8,9,10,10,11,12]` — index ≠ value. Decoding via the per-bank codebook fixes it.

Result: **a perfect bijection over all 256 × 3200 = 819,200 positions** (o max 255, i max 3199,
819200 distinct pairs).

## 4. The recovered z-order

```
within a bank:   o = 16*bank + (slot % 16)          i = slot // 16
bank b holds output channels [16b, 16b+16), all 3200 inputs
```

Exact for all 16 banks. So **16 output channels vary fastest**, then the input index increments —
the same 16-output OCG group as the 3B, but **with no input pair swap**: `i0` reads 0,0,…,0 then
1,1,…,1 across consecutive groups, whereas the 3B tile has `c = ig ^ 1` (interleave-factor-2 swap).
The interleave differs at this geometry.

Since 16 banks × 16 outputs = 16 scales per bank, the header's 16 scales are naturally **one per
output channel**, which finally accounts for a scale count that fits none of the earlier tile-shape
hypotheses.

## 5. Honest status: the map does not yet decode the shipped weights

Applying the map to pico's real down blocks and scoring against the captured logits gives, at best,
▁Paris rank 25888 / correlation **+0.052** — noise, against a depth-0 baseline of rank 2213. Every
scale-axis variant tried (`per_out`, `per_block`, `per_in`) is likewise noise.

The scale axis cannot be settled from the data: grouping the real payload by `slot%16`, `slot//3200`,
`slot//16 %16`, or `slot//1600` all give saturation 1.000, because with 3200 elements per group every
grouping contains both codebook extremes. The nibble stream is scale-normalized and therefore
statistically homogeneous by construction — this test has no power, and nothing should be read into it.

Three candidate explanations remain, in rough order of likelihood:

1. **`OutTrans=1`** changes the coefficient layout, and the probe compiles at `OutTrans=0`.
2. **The scale-bearing mode** (128-byte header) may lay the payload out differently from the
   plain-LUT mode the probe emits.
3. **gate/up's own neuron order is unverified.** The functional oracle exercises the whole FFN, so a
   correct down cannot show up while gate/up are mis-assembled. Their tiles are the `N` class
   (0x2080) and are amenable to exactly the same positional read — that is the next step, and it
   removes the confound rather than guessing around it.

## 6. What is now established

The method works and has been executed at pico's geometry: the ANE conv config was read out of
Apple's own binary, reproduced with the compiler, probed, and decoded to a verified bijection. The
z-order is no longer unknown — it is written above in closed form. What remains is not a search
problem but two identified configuration deltas and one unverified neighbouring tensor.

Artifacts: [`src/pico_downproj_zorder.py`](src/pico_downproj_zorder.py) (decoder),
[`pico_zorder_formula.json`](pico_zorder_formula.json). No Apple weights are committed.
