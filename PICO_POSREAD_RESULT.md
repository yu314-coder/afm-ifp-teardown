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

---

## 9. An mlprogram-capable ANE compiler exists — and it was already in the toolchain

§8 named the blocking capability as "an mlprogram-capable ANE compiler." One exists, it was sitting
unbuilt in this repo's own toolchain, and it works.

**`mil_to_hwx`** (`coreml_to_ane_hwx/mil/mil_to_hwx.cc`) links `ANECompiler.framework` and calls
`ANECCompile(optionsDict, flagsDict, callback)` directly on a `model.mil`, bypassing the espresso
`NeuralNetwork` loader that `coreml2hwx` is built on. `make -C mil` builds it. The working pipeline is:

```
MIL builder -> mlprogram -> ct.optimize.coreml.palettize_weights
            -> xcrun coremlcompiler compile   (handles mlpackage, incl. palettized)
            -> mil_to_hwx -a <arch>           (ANECCompile: MIL -> hwx)
```

Xcode's `coremlcompiler` compiles the palettized `.mlpackage` without complaint, and the emitted
`model.mil` confirms the composition is exactly as predicted:
`constexpr_lut_to_dense` + `constexpr_blockwise_shift_scale`.

**What this compiler accepts** (each verified end-to-end to a parseable `.hwx`):

| input | result |
|---|---|
| plain fp16 conv, iOS16 and iOS18 | compiles |
| 4-bit LUT palettization, iOS16 and iOS18 | compiles |
| `per_tensor`, `per_grouped_channel` (gs = 16, 32, 256) | compiles |
| architectures `h16`, `h17`, `h18`, and **`h16g`** | compiles |
| `per_grouped_channel` gs=1 | rejected |
| **LUT + `enable_per_channel_scale`** | **`InvalidMILProgram`** |

The arch whitelist that rejected `h16g` is `mil_to_hwx`'s own, not `ANECCompile`'s; patched behind an
`ANE_ARCH_ANY` env var, `h16g` — the architecture named in AFM's own `main-h16g.odix` — compiles fine.

**What it does not do.** No configuration produces pico's coefficient layout. Every successful
compile emits `CoeffSize[0] = 0x6440` (64-byte header), never pico's `0x6480` (128-byte header) —
across all granularities, both palettization opsets, and all four architectures including `h16g`.

**The residual difference is now exact.** Comparing the probe's ANE task against pico's shipped
down-proj task, the geometry and kernel config match completely (`InDim C=3200`, `OutDim C=256`,
`OCGSize=4`, `ActiveNE=4`, `Fmt=FLOAT16 Pal=1(4bit) SparseEn=0 Reuse=0 SBS=0 Asym=0`). Exactly three
`MacCfg` fields differ:

```
                 probe      pico
OutTrans           0          1
FillLowerNE        0          1
SmSrc              1          0
```

`FillLowerNE=1` is a plausible mechanical explanation for the header being exactly **2 x 64** bytes:
coefficients laid out to fill both NE halves would duplicate the per-bank header. This is a
hypothesis, not a demonstrated fact.

These are compiler-*chosen* fields, not exposed flags, so reaching them requires finding the graph
that makes `ANECCompile` select them. Mirroring pico's actual structure — four `3200 -> 256` convs
concatenated to 1024 and added to a residual, with and without a preceding SwiGLU
(`sigmoid`/`mul`/`mul`) — compiles cleanly but still yields `OutTrans=0, FillLowerNE=0, 0x6440` on all
four convs.

**Status.** The capability gap named in §8 is closed: MIL can be compiled to hwx, and the positional
read can be run through the mlprogram path. The *mode* gap is not: pico's shipped coefficient layout
has not been reproduced, so the read still cannot be performed in the shipped mode. The open question
is now narrow and concrete — what makes `ANECCompile` select `OutTrans=1` / `FillLowerNE=1` — rather
than "find a different compiler."

**Next lead (unexplored).** `mil_to_hwx` calls `ANECCompile` *in-process* with its own flags dict.
The production path is different: loading a model on ANE through CoreML dispatches to
`ANECompilerService.xpc` (`/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/`),
which was observed running after a `CPU_AND_NE` load and predict. That service compiles with Apple's
own flag set — plausibly the one that selects `OutTrans=1`/`FillLowerNE=1`. Its output is not cached
as a `.hwx` anywhere on disk (a system-wide scan of `/private/var` and `~/Library` found only
simulator handwriting models), and `~/Library/Caches/com.apple.e5rt.e5bundlecache` stays empty for
plain CoreML models, so capturing it would require intercepting the XPC transaction or the service's
in-memory buffer rather than reading a file.

---

## 10. The N-class positional read: ground truth for gate/up/Q/K/V/O

The remaining confound named in §5 was that gate/up's arrangement was never independently verified,
so a correct down-projection could not show through. That confound is now removed.

**N tile arithmetic.** `CoeffSize 0x2080` = 8320 B = 128 B header + 8192 B payload = 16384 nibbles;
16 banks × 16384 = 262144 = **256 out × 1024 in**. So the probe geometry is `Cin=1024, Cout=256` —
the same shape family as the L read, and the `s` class is the same with 8 outputs per bank
(`0x1080` = 128 + 4096 → 8192 nibbles, 16 banks × 8 out × 1024 in = 128 out).

**Result: a second perfect bijection.** Five digit probes compiled at that geometry decode to
**262144 distinct (o, i) pairs — a complete bijection** — and satisfy exactly the same closed form
recovered at the L geometry:

```
o = 16*bank + (slot % 16)        i = slot // 16
```

verified EXACT for all 16 banks. So the intra-tile z-order is now ground-truth-established for
**both** tile classes, i.e. for every weight role in the model.

**But it still does not decode the shipped weights.** Applying it to pico's real tensors and scoring
against the captured logits (depth-0 baseline: corr +0.0380, rank 2213):

| configuration | attention-only, layer 0 |
|---|---|
| per-output scale, head-major | corr +0.047 / rank 25710 |
| **no scale, head-major** | **corr +0.090** / rank 4484 |
| per-input-group scale, head-major | corr +0.034 / rank 62921 |
| any of the above, dim-major | corr −0.014…+0.022 |

The `no scale` line is **not** the breakthrough it appears to be. Without the per-output scales the
attention output has r.m.s. **672** against a residual of order 1, so it does not update the residual
stream — it replaces it. That correlation therefore describes the attention output in isolation, not
a functioning layer, and it does not survive: the full 24-layer forward diverges (r.m.s. 672 → 3923,
correlation −0.03). With scales applied the magnitudes are sane (r.m.s. 1.45 at depth 1) but
correlation still decays monotonically with depth (+0.048 → −0.019 by depth 24).

**Interpretation.** The z-order is correct *for the configuration the probe compiles*
(`OutTrans=0`, 64-byte plain-LUT header). pico ships `OutTrans=1` with a 128-byte scale-bearing
header. Two independent geometries now give the same closed form and neither reproduces the shipped
weights, which strengthens rather than weakens the conclusion of §8–§9: the residual difference is
the **compiled mode**, not the tile geometry or the element order within a mode.

The gate/up confound is eliminated — their arrangement is now known on the same footing as the
down-projection — and the pico forward still does not work. That localises the remaining error to
the mode difference alone.

---

## 11. Round-trip validation of the decoder, and the definitive blocker

Two clean results this pass settle where the pico forward actually stands.

**The decoder is provably correct (for the compilable mode).** Compiling a conv with *known* random
fp16 weights at `Cin=1024, Cout=256` (the mode `mil_to_hwx`/`coreml2hwx` can produce, `OutTrans=0`,
64-byte plain-LUT header) and decoding its `__kern_0` stream with the recovered z-order gives:

```
codebook[nibble] decode, no scale:   correlation 0.981 vs true weights
```

0.981 is the 4-bit palettization floor — the z-order and codebook decode are **exact up to
quantization**. (Applying a "scale" from bytes [64:96] here drops it to 0.51, because this compiled
mode has no scale table there; that is a property of the *mode*, not a decoder error.)

**The scale location is confirmed for shipped tiles.** pico's real `0x6480` headers carry the
codebook at fp16[0:16] and a genuine per-output scale at **fp16[32:48]** (= bytes [64:96]): Q reads
0.089–0.257, down reads 0.099–0.443, and the `s` class has exactly 8 plausible values there
(matching 8 outputs/bank) before garbage. The decoder already reads this slice. So for shipped
weights the decode is `codebook[nibble] · scale[output]` with `o = 16·bank + slot%16, i = slot//16`,
and every component of that is now independently validated.

**Therefore the forward's failure is not the decoder.** With the decode confirmed, the residual gap
is the one difference that remains between the probe and the shipped tiles: **the shipped down-proj
runs `OutTrans=1`** (four consecutive `3200→256` conv tasks fed by the SwiGLU `Mul`, all
`OutTrans=1`), while every conv this toolchain can compile is `OutTrans=0`.

**`OutTrans=1` is not reproducible with the available tooling.** It is a graph-scheduling decision,
not a shape or flag: sweeping conv shapes `(Cin,Cout,S)` ∈ {(1024,256,64), (3200,256,64),
(1024,1024,64), (256,1024,64), (1024,256,{1,256}), (64,64,1024), (256,256,64), (2048,256,64)}
through `mil_to_hwx` yields `OutTrans=0` in every case, and reproducing pico's `Mul→conv` fragment
also compiles to `OutTrans=0`. The transpose is chosen by the ANE scheduler from the full-graph
context (what consumes the output), which cannot be recreated from an isolated op.

**Weak structural lead.** Under the scale-decontaminated low-rank statistic R (genuine weights ≫
noise), the FFN roles score *higher* transposed — `gate` R 6.0 vs 4.9, `down` R 10.4 vs 5.8 — while
`Q` prefers the untransposed order (10.2 vs 7.6). This hints the `OutTrans=1` coefficient order for
the FFN tiles is closer to a within-bank transpose, but R rewards structure that a transpose
preserves, so it is suggestive only; the functional oracle does not confirm the transpose (§10).

## 12. Honest status of the pico (300M) reconstruction

**Component-complete and validated:** embedding (bit-exact, semantically validated), tied unembed
(depth-0 ranks ▁Paris 2213/262000), all weight *values* and the intra-tile z-order (round-trip
0.981, both tile classes, closed form `o=16·bank+slot%16, i=slot//16`), the per-output scale
location, the norms (γ folded, QK-norm unit), the true input (lowercased, chat-templated), and the
functional oracle (captured full logits).

**Not achieved:** a coherent from-weights forward. It degrades monotonically with depth from the
depth-0 baseline, and the sole unresolved variable is the **`OutTrans=1` coefficient ordering of the
shipped tiles**, which cannot be read because no available compiler emits `OutTrans=1` for an
isolated conv, and pico's own activations are ANE-internal (so it cannot be captured at runtime
either). This is a genuine tooling/observability wall, not a remaining search: the decoder is proven,
the geometry is proven, and the missing piece is one scheduler-chosen storage transpose that the
shipped assets exercise but the reproduction path does not.
