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

---

## 7. The bank interpretation: structure confirmed, forward still broken

The positional read forces a reinterpretation of the whole pico weight map, and the arithmetic closes
exactly for every tensor — but it does **not** yet produce a working forward. Both halves of that
sentence matter.

### What the structure is

A "tile" is not a 2D spatial tile. It is an ANE **coefficient bank** holding a fixed number of output
channels across **all** inputs, with the block being one ANE task:

```
N bank: 16384 nibbles = 16 out x 1024 in, 16 scales
s bank:  8192 nibbles =  8 out x 1024 in,  8 scales
L bank: 51200 nibbles = 16 out x 3200 in, 16 scales
```

This resolves several things that no earlier tile-shape hypothesis could:

- **gate/up's `sNNNNNNNNNNNN` block structure**: 12 N blocks x 16 banks x 16 out = 3072, plus one
  s block x 16 banks x 8 out = 128, giving exactly **3200**.
- **why the `s` class has 8 scales and not 16**: 8 output channels per bank.
- **the L class's 16 scales**, which fit none of the 20 candidate tile shapes tried earlier.
- Every one of the seven roles builds to exactly its expected output count
  (Q 1024, K 256, V 256, O 1024, gate 3200, up 3200, down 1024), with consistent magnitudes
  (rms 0.028-0.040).

Independently, Apple's own working 3B decoder (`afm_odix/build_model_state.py`) applies the per-1024
fp16 scale **in raw index order before de-swizzling**, then `W = v.reshape(rows//8, Ci, 8).transpose(0,2,1)`.
For a pico N block (256 out x 1024 in = 262144 elements) that is 256 scale groups — exactly the
16 banks x 16 scales present. The scale accounting closes.

### What still does not work

None of it moves the oracle. Scored against the captured logits (depth-0 baseline: corr +0.0380,
rank 2213):

| assembly | attention only | FFN only |
|---|---|---|
| bank interpretation, per-output-channel scale | +0.047 / 25710 | −0.017 / 201688 |
| 3B scheme, per-1024 raw-order scale, IF=8 | +0.026 / 104288 | −0.003 / 38784 |
| 3B scheme, IF=16 | +0.020 / 112132 | +0.027 / 199737 |
| deswizzle sweep (OB in {8,16} x IB in {64,128}) | +0.025..+0.039 / 24307..177240 | −0.027..−0.010 / 129656..216880 |

Every entry is inside the noise band (|corr| < 0.06). Notably the bank interpretation makes
*attention* worse than the earlier spatial-tile decode did, which is evidence against the specific
index mapping even though the counts close.

### Honest assessment

The structural claim (blocks = ANE tasks, tiles = coefficient banks, counts closing exactly for all
seven roles) is well supported and independently corroborated by the hwx config. The **ordering**
within that structure is not solved: the recovered z-order is bit-exact for the config compiled at
`OutTrans=0` with a plain-LUT header, and pico ships `OutTrans=1` with a scale-bearing header.

A methodological note on why this is slow to converge: the functional oracle exercises the *entire*
block at once — embedding, seven tensors, head layout, norms, RoPE, residual — so any single wrong
element masks every other correct one, and §3 established there is no per-layer ground truth to
bisect against. Enumerating whole-block configurations is therefore a poor search strategy, and the
results above should be read as a record of what was excluded, not as progress toward a fit.

---

## 8. The `ct.optimize.coreml` palettization route is closed (toolchain limit)

§5 identified the scale-bearing compile mode as the most promising way to reproduce pico's exact
`CoeffSize 0x6480` header and re-run the positional read in the true mode. The modern
`coremltools.optimize.coreml` API does expose it — `OpPalettizerConfig` has
`enable_per_channel_scale`, i.e. LUT + per-channel scale, exactly pico's
codebook + 16 fp16 scales layout — but it only applies to **mlprogram** models.

Building the conv through the MIL builder, converting to mlprogram, and palettizing all work
(`per_tensor` and `per_grouped_channel`, with and without per-channel scale, all produce saved
`.mlpackage`s). **`coreml2hwx` then fails on every one** with:

```
espresso_plan_add_network ret -1
```

The decisive control is a **plain, unpalettized mlprogram**, which fails identically. So this is not
a palettization problem — `coreml2hwx` cannot consume mlprogram at all. It is a NeuralNetwork-only
harness around the legacy espresso plan loader, and the legacy `NeuralNetwork` format cannot express
this mode either: `WeightParams.quantization` carries *either* `linearQuantization` *or*
`lookupTableQuantization`, never both, whereas pico's tile needs a lookup table **and** a per-channel
scale (in mlprogram terms `constexpr_lut_to_dense` composed with `constexpr_blockwise_shift_scale`).

**Consequence.** The probe can only ever be compiled in the plain-LUT mode (`0x6440`, 64-byte header),
so the positional read cannot currently be performed in pico's shipped mode. Reproducing
`OutTrans=1` *and* the scale-bearing header requires an **mlprogram-capable ANE compiler** — a
different tool than the one in this repo's toolchain. That is now the concrete blocking capability,
rather than an open search problem.

Environment note: this required coremltools with working native extensions
(`libmilstoragepython`); the Python 3.14 install is pure-Python and raises `BlobWriter not loaded`
on mlprogram save. Python 3.12 with the `cp312-none-macosx_11_0_arm64` wheel works (note the symbol
is `_BlobStorageWriter` in v9.0, not `_BlobWriter`).
