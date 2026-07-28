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

---

## 13. What `OutTrans=1` actually selects: the residual-writing ops

Parsing every conv task in pico's own `binary_0.hwx` (1003 tasks with a `CoeffSize`) and grouping by
shape and mode gives a clean decomposition. Every projection is emitted as `→256` output chunks:

```
InC    OutC   OutTrans  CoeffSize   count
1024   256    0         0x2080       546
1024   256    1         0x2080        72     = 18 layers x 4
3200   256    1         0x6480        72     = 18 layers x 4
1024   128    0         0x1080        36
```

Reading the task sequence within one layer identifies them exactly. Tasks 174–177 are four
`1024→256` `OutTrans=1` convs immediately after the attention softmax/reshape ops — the **O
projection** (`1024→1024` split into 4 chunks). Tasks 218–221 are the four `3200→256` `OutTrans=1`
convs immediately after the SwiGLU `Mul` (task 217) — the **down projection**. Tasks 184–209, the
gate/up matrices (`1024→3200` = 12×256 + 1×128), are all `OutTrans=0`.

**The rule is architectural:**

> `OutTrans=1` is used by exactly the ops that **write into the residual stream** — the attention
> output projection and the FFN down projection (plus their LoRA up-projections, the `32→1024`
> `OutTrans=1` tasks). Everything that reads from the normed hidden state — Q, K, V, gate, up — is
> `OutTrans=0`.

**Consequence: most of pico is already decoded correctly.** `OutTrans=0` is precisely the mode the
positional read validated (round-trip 0.981), so **Q, K, V, gate and up are correct as decoded
today**. Only the two residual-writing roles carry the unknown ordering. That shrinks the open
problem from "all seven roles" to two, and from "an arbitrary intra-tile order" to what is most
likely an output-channel permutation.

**Mixed-mode results.** Decoding Q/K/V/gate/up with the validated order and varying only O/down:

| O/down order | depth 1 | depth 6 | depth 24 |
|---|---|---|---|
| `ofast` (as OutTrans=0) | +0.0475 / 51875 | −0.0081 / 97114 | −0.0185 / 158351 |
| `ifast` (intra-bank transpose) | **+0.0811** / 90127 | +0.0212 / 131657 | −0.0039 / 203113 |
| `obank` (bank-transposed output map) | −0.0009 / 39452 | +0.0119 / 15704 | +0.0234 / 28104 |
| `ifast_obank` (both) | +0.0365 / 122117 | +0.0479 / 19598 | +0.0275 / **3749** |

Two things improve measurably relative to the uniform decode: `ifast` doubles the depth-1 correlation
(+0.081 vs +0.047) at identical scaling, so it is not a magnitude artifact; and `ifast_obank` is the
first configuration whose **full 24-layer** forward does not collapse — rank 3749 against the depth-0
baseline of 2213, where the uniform decode reaches 158351.

**But this is not a solve.** Every correlation remains inside the ±0.06 noise band established
earlier, and no configuration exceeds the depth-0 correlation of +0.0380. The honest reading is that
restricting the unknown to the residual-writing roles is a real structural advance, and that the
specific output permutation is still unidentified.

**Separate observation worth recording.** The same trace shows pico's graph carries **rank-32 LoRA
adapters** inline: `1024→32→1024` around the O projection (tasks 171–173), `1024→32→3200` + `Add`
for gate and for up (210–215), and `3200→32→1024` for down (222–223). The base asset's 998 weight
blocks are fully accounted for by the 168 base tensors, so these adapter *weights* live in the
separate `lora_32`/`lora_48` assets rather than the base file — but if the captured-logit oracle was
recorded with an adapter resident, the reconstruction is missing an additive term that no weight
ordering can compensate for. That is an untested confound on the oracle itself.

---

## 14. The LoRA confound: resolved negative

Finding §13 raised the possibility that the captured-logit oracle was recorded with a LoRA adapter
resident, which would leave the reconstruction missing an additive term that no weight ordering could
compensate for — making the target unattainable by construction. **That is not the case.**

pico's own `metadata.json` declares its adapter slots explicitly:

```json
"backbone_signature": "cc4da08ebb47cce3de0d53aa90ba5453639c06db28f487d395c72f2fff4196cf",
"adapter_type_to_signature_mapping": {
    "lora_32": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "lora_64": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

Both signatures are **SHA-256 of the empty string** (verified directly). The same value appears as
`adapter_signature` in **47 of the 49** adapter assets in the catalog, including the `lora_32` asset
whose `backbone_signature` matches pico's exactly — i.e. the adapter that targets this model.

So the graph carries LoRA *slots* (the `1024\to32\to1024` and `1024\to32\to3200` conv pairs visible
in the task trace) but no adapter *data*. The captured logits therefore reflect base weights alone,
the reconstruction is not missing an additive term, and the oracle remains a valid target. The
confound is closed, and the forward failure stays attributable to the `OutTrans=1` ordering.

A second, independent check agrees: pico's weight file contains **998** coefficient blocks, and the
base architecture accounts for exactly **960** of them (24 layers × 40 = Q 4 + K 1 + V 1 + O 4 +
gate 13 + up 13 + down 4). No block budget remains for LoRA weights.

## 15. Thirty-eight unaccounted blocks: a layer-shaped unit after layer 23

The same audit surfaces something not previously noticed. The 38 blocks beyond the base 960 sit in
the weight map's `PARTIAL_UNIT` entry, and they lie **after every mapped layer** (offset range
`0xb327040`–`0xb889cc0`, where layer 23 ends at `0xb2c0800`). Their class composition is

```
32 N + 2 s + 4 L
```

which is exactly one layer (40 blocks) **minus K and V** (one N block each). Splitting them by file
order against a normal layer's block order confirms it: gate, up and down assemble to the **exact**
expected shapes — `(1024, 3200)`, `(1024, 3200)`, `(3200, 1024)` — leaving 8 N blocks for Q and O.
This is a complete, correctly-shaped transformer layer without its own K/V projections, consistent
with a **KV-reusing** layer (the `odix` architecture notes already record that some pico layers reuse
KV).

Running it as a 25th layer, with K/V carried over from layer 23, changes the result but does not fix
it (correlation −0.019 → +0.002, rank 158351 → 124513): still noise, as expected while the ordering
is unresolved everywhere. So this is recorded as a **structural** finding, not a functional one — but
it means both the from-weights forward and the GGUF export currently omit a layer-shaped unit that
the shipped model contains, and the "24 dense layers" figure derived from the `odix` op census should
be treated as provisional.

## 16. `OutTrans=1` graph-context hypothesis: tested and ruled out

Every prior `mil_to_hwx` sweep compiled a **bare, isolated** convolution (varying only `cin`/`cout`/`S`)
and always got `OutTrans=0`. It remained an open question whether `OutTrans=1` is a graph-*context*
effect -- i.e. whether the ANE scheduler only chooses it when the conv's output is consumed by a
subsequent op (a residual add), or when preceded by the same op that precedes it in pico's real graph
(a softmax for O, a SwiGLU gate for down), rather than by shape alone. Two things blocked testing this
before: (1) `coreml2hwx`, the tool used to reach that graph-shape hypothesis via the legacy
`NeuralNetworkBuilder` API, cannot load a modern ML Program `.mlpackage` at all (`espresso_plan_add_network
ret -1`) -- so an earlier attempt at exactly this test never actually exercised the compiler; (2) the
working `mlprogram -> xcrun coremlcompiler -> mil_to_hwx` pipeline was only ever driven with a bare conv.

Combining the two -- real graph shapes through the pipeline that actually loads ML Programs -- gives a
real test, run this session (`outtrans_residual.py`). Compiled and read via `mil_to_hwx`/`hwx_parsing`:

| variant | preceding op(s) | following op | `OutTrans` |
|---|---|---|---|
| bare conv | none | none | `0` |
| `mul -> conv` | generic elementwise mul | none | `0` |
| `conv -> add` | none | residual add | `0` |
| `mul -> conv -> add` | generic mul | residual add | `0` |
| 2x stacked `mul -> conv -> add` | same, repeated, sharing one residual stream | residual add (both blocks) | `0` |
| `softmax -> conv -> add` | **exact** O-proj predecessor (task order: softmax then conv) | residual add | `0` |
| `silu(gate) * up -> conv -> add` | **exact** down-proj predecessor (real SwiGLU, not a generic mul) | residual add | `0` |

Every one of these compiled cleanly and reports `OutTrans=0`, across every weight-palettization
granularity the local ANE compiler will accept (`per_tensor`, `per_grouped_channel` with
`channel_axis=0`). This **rules out** local graph context (residual writes, 2-layer repetition, and
exact real predecessor op identity) as the trigger, closing a hypothesis that earlier writeups left
open pending a working test.

One config remains genuinely untestable, not merely unexplored: `enable_per_channel_scale=True` (or
`per_grouped_channel` with `channel_axis=1`) is the palettization scheme that actually matches pico's
real container (per-output-channel scale), but it lowers to `constexpr_blockwise_shift_scale`, an op
this machine's `mil_to_hwx`/ANE compiler rejects outright (`InvalidMILProgram`) before it ever reaches
the scheduler stage that assigns `OutTrans`. This is unlikely to matter -- `OutTrans` governs
*activation* storage, not weight coefficient encoding, and every compilable weight scheme gave the
same null result -- but it means the palettization axis of the search space is blocked by tooling, not
resolved.

Net effect: the `OutTrans=1` trigger is not a local, few-op graph property at all. What was left
untested was real **model-scale** context -- addressed next.

## 17. Full 24-layer depth sweep: also ruled out

Built `outtrans_depth.py`: N repeated blocks (`softmax -> conv_O -> add` then
`silu(conv_gate) * conv_up -> conv_down -> add`) at pico's real widths (`D=1024`, `FFN=3200`),
chained through **one continuous residual stream**, at `n_layers` = 1, 2, 4, 8, 16, 24 -- i.e. the
full real depth, with every `O` and `down` conv sharing the same growing whole-program graph a real
compile of pico would see. All six depths compiled cleanly through the same
`mlprogram -> coremlcompiler -> mil_to_hwx` pipeline:

| n_layers | conv tasks | `OutTrans` values seen |
|---|---|---|
| 1 | 14 | `{0}` |
| 2 | 27 | `{0}` |
| 4 | 53 | `{0}` |
| 8 | 105 | `{0}` |
| 16 | 209 | `{0}` |
| 24 | 313 | `{0}` |

`OutTrans=1` never appears, at any depth, for any of the 313 conv tasks in the full-depth graph. This
rules out repetition depth / whole-program size as the trigger too -- the entire "reproduce it via a
synthetic MIL graph, at any shape, any local context, or any depth" avenue is now exhausted.

What is NOT reproduced in this test (and remains the one un-eliminated fidelity gap) is the real
multi-head **reshape/transpose** pattern: pico's true attention path is
`softmax(QK^T) -> matmul(V) -> reshape/transpose (merge heads back to channels) -> conv(O)`, whereas
this test fed `softmax` directly into `conv(O)` with no intervening transpose. It is possible the
scheduler's `OutTrans=1` choice is keyed specifically off a transpose op sitting immediately upstream
of the conv (cheap to fuse into a storage transpose) rather than off softmax/depth/residual-writes at
all -- a hypothesis distinct from everything tested above, and not yet tried.

## 18. Transpose adjacency DOES emit `OutTrans=1` -- but only on weightless tasks

`outtrans_transpose.py` tests the #17 hypothesis. Adding a real head-merge transpose upstream of the
conv produces the **first `OutTrans=1` ever obtained from a synthetic graph** in this project:

| variant | `OutTrans` values | conv tasks |
|---|---|---|
| `transpose -> transpose -> conv` | `{0}` | 1 |
| `reshape -> conv` | `{0}` | 2 |
| `transpose -> reshape -> conv -> add` | **`{0,1}`** | 3 |
| full real MHA (QKV convs, head reshape, `QK^T`, softmax, `V` matmul, head-merge transpose) `-> conv_O -> add` | **`{0,1}`** | 16 |
| 4x stacked real MHA blocks | **`{0,1}`** | 61 (8 of them `OutTrans=1`) |

So transposes are unambiguously *involved* in `OutTrans=1`. **However, this does not reproduce pico's
condition.** Parsing which task carries the flag:

* In the synthetic graphs, every `OutTrans=1` task is **weightless** -- `Pal=-`, `banks=0`, no
  `CoeffSize` -- i.e. the compiler emits the head-merge transpose as its *own* standalone shuffle task
  and leaves the consuming palettized conv at `OutTrans=0`.
* In pico, `OutTrans=1` sits on **weight-bearing convs**: of 363 `OutTrans=1` tasks, **180 carry
  coefficient banks** (`Pal=1(4bit)`, 15 `CoeffBase` entries, real `CoeffSize`).

The transpose is therefore *fused into* the conv's coefficient read in the real model, and *not fused*
in every graph the local compiler will build. Reproducing `OutTrans=1` on a **weight-bearing** conv --
the only kind whose coefficient stream a positional read could decode -- remains unachieved.

### A shape-independence result that supersedes the shape sweeps

The census of pico's own weight-bearing convs settles a question the earlier shape sweeps could not:

| Cin -> Cout | CoeffSize0 | `OutTrans=0` | `OutTrans=1` |
|---|---|---|---|
| 1024 -> 256 | `0x2080` | **546** | **72** |
| 3200 -> 256 (down-proj) | `0x6480` | 0 | **72** |
| 32 -> 1024 | `0x1000` | 19 | 36 |
| 1024 -> 32 / 32 -> 3200 / 1024 -> 128 / ... | various | 808 total | -- |

The identical configuration `1024 -> 256, CoeffSize 0x2080` occurs **both ways** (546 vs 72). This is
direct proof from Apple's own binary that `OutTrans` is **not a function of conv shape or coefficient
container** -- it is a scheduler decision made from surrounding graph context. Every shape-sweep
approach was therefore searching a space that provably cannot contain the answer.

It also *confirms* the residual-write rule stated in #13: the down-projection is `OutTrans=1` in
**72 of 72** instances (never 0), and the `OutTrans=1` subset of the `1024 -> 256` population numbers
exactly **72** as well -- matching down-proj's count, consistent with these being the O-projections.
`Q`/`K`/`V`/`gate`/`up` make up the `OutTrans=0` remainder. So "the two projections that write into
the residual stream are the `OutTrans=1` ones" survives this much sharper test.

No readable config field separates the two populations (diffing every `key=value` in both groups
returns only incidental address/`Tag` differences), consistent with the flag being assigned by the
scheduler rather than derived from any property recorded in the task.

## 19. A LoRA Rosetta stone: `OutTrans` does **not** change coefficient order

### The unbacked-segment discovery

Dumping the hwx segment table (`hwx_parsing -s`) shows two segments with **`File Size 0`** -- allocated
in VM but not backed by any bytes in the file:

| segment | VM size | = |
|---|---|---|
| `__MKERN_0` | `0x1c50000` = **29,687,808** | `lora_32_constant_data.bin` = **29,687,808** exactly |
| `__MKERN_9` | `0x38a0000` = **59,375,616** | `lora_64_constant_data.bin` = **59,375,616** exactly |

The sizes match exactly, so the constant-data files are DMA'd **verbatim** into ANE coefficient
memory. Their bytes therefore *are* the ANE coefficient stream, already in tile order -- the same
runtime-patched mutable-kernel pattern recorded for the 3B.

This is a Rosetta stone, because the LoRA tensors are **unpalettized fp16** (task 157 carries no `Pal`
field; `CoeffSize 0x1000 x 16 banks = 65536 B` for `32x1024` = exactly 2 B/element), so there is no
quantization loss, **and both `OutTrans` modes appear in the same file**: the `32 -> 1024`
up-projections are `OutTrans=1` (36 tasks) while `32 -> 3200` are `OutTrans=0` (36 tasks).

### File layout (derived exactly, zero residual)

`14,843,904 fp16 = 24 layers x 618,496`, and the rank-32 adapters for all seven roles sum to exactly
618,496 per layer: `Q/O` `1024*32 + 32*1024`, `K/V` `1024*32 + 32*256`, `gate/up` `1024*32 + 32*3200`,
`down` `3200*32 + 32*1024`. Tensor boundaries were then confirmed empirically from log-RMS
discontinuities (9 of the derived boundaries land exactly on measured jumps; the `K`/`V` sub-region is
the one part that does not resolve cleanly and was excluded from the test).

### The test and the result

A `32 -> out` tensor is stored as 16 banks; each bank holds `out/16` output channels x 32 ranks. The
**rank axis is global** (the same 32 latent directions in every bank) while the **output axis is local**
(different channels per bank). So under the *correct* intra-bank order the per-rank norm profile
extracted from each bank measures the same quantity in every bank and correlates across banks; under
the wrong order it mixes output channels and decorrelates.

Method validated on the **known** `OutTrans=0` mode first, then applied to `OutTrans=1`:

| tensor | mode | `in_fast` | `out_fast` | outcome |
|---|---|---|---|---|
| `gate_B` (32->3200) | `OutTrans=0` (known) | mean **+0.125**, positive 21/24 | ~0 | validates the method |
| `Q_B` (32->1024) | **`OutTrans=1`** | mean **+0.088**, positive 23/24 | −0.009, positive 7/24 | `in_fast` wins **24/24 layers** |
| `down_B` (32->1024) | `OutTrans=1` | +0.031, positive 10/24 | −0.001 | **ambiguous**, 12/24 |

`Q_B` is boundary-confirmed and behaves exactly like the known `OutTrans=0` tensor, at comparable
signal strength. `down_B` does not separate and is reported as inconclusive rather than as support.

### What this means

> On this evidence **`OutTrans` does not change the intra-bank coefficient order.** It is an
> **output-activation layout** flag -- which is what the name says, and what the synthetic
> reproduction independently showed in #18: every `OutTrans=1` task generated there was a *weightless*
> shuffle with identical `InDim`/`OutDim`, i.e. a task that moves activations, not weights.

That makes the long-standing "`OutTrans=1` coefficient ordering" framing a likely **misdiagnosis**. If
the flag never permutes coefficients, then `O` and `down` are already decoded with the correct order
and the incoherent forward has a different cause.

**Caveat, stated plainly:** the Rosetta tensors are unpalettized fp16 while the base `O`/`down` tiles
are 4-bit palettized with per-output scales, so this transfers the *flag semantics*, not a
byte-identical layout proof for the palettized path. `down_B` being inconclusive leaves room for the
result not to generalize across every role. This is strong evidence, not a closed proof.

### Where the search should go next

With `OutTrans` demoted, the prime suspects become the **arrangement conventions** already enumerated
as knobs in `src/pico_forward.py` -- each a small discrete choice, jointly a far smaller space than an
unknown permutation:

* `kv_mode` -- the single 512x512 K/V block -> `[1024,256]` (`reshape` vs `topslice`)
* `s_position` / `s_tile_shape` -- where the `s` half-block sits in `gate`/`up` (`[1024,128]` columns)
* `rope_interleaved` -- rotate-half pairing (`#4` of `PICO_NORMS.md` fixes rotate-half; the pairing
  convention in the reconstruction is still a choice)
* head ordering / GQA fan-out
* the **38-block KV-reusing 25th layer** (#15), currently omitted from both the forward and the GGUF

## 20. Residual-basis test: the defect is the O/down **output** ordering (a fast static oracle)

`O` and `down` **write** the residual stream; `Q/K/V/gate/up` **read** it. All five readers decode in
the round-trip-validated order, so their INPUT axis is a trustworthy sample of the residual basis. If
the writers' OUTPUT axis were correctly ordered it would describe the same per-position structure.

| measurement | mean over 8 layers |
|---|---|
| **calibration** `gate_in ~ up_in` (two known-correct readers) | **+0.6501** |
| `O_out ~ gate_in` (writer vs reader, same layer) | **+0.0112** |
| `down_out ~ Q_in(L+1)` (writer vs next layer's reader) | **+0.0219** |
| randomly shuffled control | +0.0034 |

The writers are **statistically indistinguishable from a random permutation** against a basis the
readers agree on at +0.65. This localises the defect precisely: it is the **output-channel ordering of
`O` and `down`**, i.e. the `ob + b*nout + o` global output index in the decoder (which
block/bank group of 16 lands where) -- *not* the intra-bank order, consistent with #19.

It also supplies what the project has lacked: a **fast, purely static oracle** with a large signal gap
(0.65 vs 0.003) that needs no forward pass, no captured logits, and no runtime. Any candidate ordering
must lift `O_out ~ gate_in` toward +0.65.

### Attempted solve, and why it failed

The permutation maximising the correlation of two 1-D vectors is rank-matching, so the permutation was
fitted by rank-matching grouped norm profiles against `gate`, then scored on readers **excluded from
the fit** (`up`, and next-layer `Q`):

| group size | fit (gate) | held-out `up` | held-out `Q(L+1)` | random-perm baseline |
|---|---|---|---|---|
| 16 (64 groups) | +0.97 | +0.331 | **+0.028** | +0.057 |
| 64 (16 groups) | +0.97 | +0.112 | +0.014 | +0.100 |
| 256 (4 groups) | +0.93 | +0.144 | −0.273 | −0.184 |

**This is a negative result.** The fitted correlation (+0.97) is meaningless -- rank-matching attains
it by construction. The apparent +0.331 on `up` is not evidence either: `up` is itself correlated with
the fit target `gate` at +0.65, so a permutation tuned to `gate` transfers to `up` automatically. The
honest read is the genuinely independent reader, next-layer `Q`, which scores **+0.028 -- exactly the
random baseline** at every granularity.

Conclusion: **the per-channel norm profile localises the defect but does not identify the
permutation.** A scalar per channel is too little information to pin a 64-way (or finer) assignment;
recovering it needs a statistic that uses the weight *vectors*, not their magnitudes -- for example
matching the writer's output directions against the reader's input directions in the shared residual
subspace, which is the natural next attempt.

## 21. Second-order statistic: premise holds, but free and structured searches both fail

The second-order upgrade is the group-level **direction-similarity** matrix `C[g,h]` (cosine between
the mean weight direction of residual groups `g` and `h`) -- a 64x64 matrix rather than 64 scalars.

**Premise test (passes).** Second-order structure transfers robustly between roles and across layers:

| | mean |
|---|---|
| `gate ~ up` (two known-correct readers) | **+0.680** |
| `gate ~ Q` | +0.527 |
| `gate ~ gate(L+1)` (across layers) | **+0.724** |
| shuffled control | −0.006 |
| `O_out ~ gate` (the writer, as decoded) | **+0.130** |

The writer scoring +0.130 rather than ~0 says the decode is *partially* aligned, not fully scrambled.

**Free-permutation QAP (fails).** A first run appeared to reject the hypothesis but was invalid: the
solver returned permutations scoring *below* the identity (+0.070 vs +0.137), because scipy's QAP
maximises a raw inner product while the target is the off-diagonal *correlation* -- with an
all-ones diagonal and an uncentred mean, the surrogate optimum is not the target optimum. After
zeroing the diagonal, centring, and seeding a restart with the identity, the solver **never beat the
identity on any layer**. So the group-level assignment is not an arbitrary permutation.

**Structured-layout enumeration (fails).** The output index is built from three fields
(`blk` x4, `bank` x16, `o` x16), so every digit ordering and per-field reversal -- 3! x 2^3 = 48
candidates -- was tested exhaustively. A first pass appeared to find a winner (`o/bank/blk`,
+0.28 vs +0.10) but that was a **grouping confound**: it grouped channels in 16s *after* permuting,
so the score partly rewarded "forms coherent groups". Re-run per channel (where a permutation is an
exact symmetric reindex `C[perm][:,perm]`), the ranking collapses:

| | 2nd-order | 1st-order |
|---|---|---|
| reference (two known-correct readers) | **+0.117** | **+0.729** |
| best of all 48 structured layouts | +0.008 | +0.019 |
| current decode | +0.002 | +0.029 |
| shuffled control | −0.003 | −0.033 |

**No digit-reordering of `(blk, bank, o)` recovers the ordering.** Both the free and the structured
hypothesis classes are eliminated.

## 22. `O` is stored **transposed** -- confirmed on two independent metrics, but it does not fix generation

Testing a different *kind* of hypothesis -- not a permutation but an axis swap -- gives the clearest
positive signal of the whole investigation. Comparing O's two axes against the residual basis:

| | 1st-order (norms) | 2nd-order (directions) |
|---|---|---|
| O **columns** as residual (what the decoder assumes) | +0.015 | +0.003 |
| O **rows** as residual (i.e. O stored transposed) | **+0.336** | **+0.057** |
| reference (two known-correct readers) | +0.700 | +0.117 |

Both metrics are independent, both agree, and the effect holds in **all 8 layers tested**, growing
with depth (1st-order: L0 +0.18 -> L4 +0.52). Transposing recovers **~48% of the reference on both
scales** (0.336/0.700 and 0.057/0.117). This is also the reading the flag name most directly
supports: `OutTrans` = *output transpose*, set on exactly the two roles whose decode this corrects.

**But it does not produce a working model, and that must be stated plainly.** Rebuilding the GGUF with
`O` transposed (`afmplus-v11.0-pico-qwen3-Ot-F16.gguf`) makes generation *degenerate* rather than
better: the model emits EOS immediately, and with `--ignore-eos` it produces empty output, versus
word-salad for the untransposed build.

Two readings, not yet separated:
1. the transpose is right for `O` but `down` -- which is `[3200,1024]` and so cannot be transposed in
   the same dimension-preserving way `O` (square) can -- remains wrong, and a half-corrected residual
   path is not better than a uniformly wrong one;
2. the static oracle, though well-controlled, is an imperfect proxy for functional correctness.

Until those are separated, the transpose is recorded as a **statistically solid structural finding
that has not been shown to improve the forward**, and the shipped default keeps the untransposed
decode. It is not presented as a solve.

### Independent confirmation via the attention-head signature

The transpose conclusion above rests on the residual-basis oracle. A second test avoids that oracle
entirely. One of O's axes is the attention head-concat (16 heads x 64), and head structure is
detectable: dimensions inside the same 64-wide head block are more alike than dimensions in different
heads. Calibrating on tensors whose orientation is *not* in doubt (`Q` = `[residual-in, head-out]`,
`gate` = `[residual-in, ffn-out]`, unambiguous from its non-square shape):

| axis | 64-block head score |
|---|---|
| `Q` axis1 -- known **head** axis | **+0.02171** |
| `Q` axis0 -- known residual axis | −0.00052 |
| `gate` axis0 -- known residual axis | −0.00012 |
| **`O` axis1** | **+0.01950** -> head |
| **`O` axis0** | −0.00152 -> residual |

`O` axis1 reaches 90% of the calibrated head score while `O` axis0 matches the residual calibration.
Two methods sharing no assumptions agree, so **O being stored transposed is established**, even though
it does not by itself repair generation.

## 23. `down` is misaligned on BOTH axes -- deeper than an ordering problem

If O were the only fault, correcting it should have helped; it did not, so `down` was examined
directly. Unlike O, `down` is `[3200, 1024]` and cannot be transposed dimension-preservingly -- the
3200 axis must be the FFN input and the 1024 axis the residual output.

Its two axes can be checked against *different* references:

| `down` axis | reference it should match | measured | reference value |
|---|---|---|---|
| 3200 (FFN) | `gate`/`up` OUTPUT axis (same FFN neurons) | **−0.044** | +0.222 |
| 1024 (residual) | `gate` INPUT axis (residual basis) | **+0.022** | +0.700 |

Both axes appear to fail. The tempting inference -- `colnorm(down)` is a norm *over* the input axis so
it is invariant to input permutation, and symmetrically for the other axis, therefore no single-axis
permutation can produce both failures, therefore the whole "wrong output order" family is eliminated
and the intra-bank slot decomposition is indicted -- **does not hold. It is retracted below.**

> ### RETRACTION: the FFN-axis reference is not a ground truth
>
> That argument requires the FFN-axis comparison to be trustworthy. It is not. The reference itself
> (`gate ~ up` on the 3200 axis) is only **+0.222**, against **+0.700** for the residual axis -- weak
> enough to deserve an independent check, which it fails.
>
> `lora_down_probe.py` supplies that check. The LoRA `down` adapter's A-matrix is `[3200, 32]`, its
> 3200 axis indexes the *same* FFN neurons, its ANE bytes are known **verbatim** (#19), it is
> unpalettized fp16, and it is in the **known-good `OutTrans=0` mode** -- so it should align with
> gate/up if that axis is meaningful at all. Measured over 6 layers:
>
> | | mean |
> |---|---|
> | `down_A` (ofast) ~ `gate` FFN axis | **+0.024** |
> | `down_A` (ifast) ~ `gate` FFN axis | +0.006 |
> | `down_A` (ofast) ~ `up` FFN axis | +0.007 |
> | reference `gate ~ up` FFN axis | +0.222 |
>
> A tensor that is *not* in question fails the same test base `down` fails. Three tensors that all
> index the same 3200 FFN neurons do not agree with one another. So the FFN axis does not carry the
> strong shared structure the residual axis does -- most likely because per-neuron magnitudes of
> `gate` and `up` have no reason to track each other under SwiGLU (`silu(gate_j) * up_j`), making
> +0.222 an artefact of weak intrinsic structure rather than a basis.
>
> **Consequences:** (1) the claim that `down`'s *input* axis is misordered is **withdrawn** -- it was
> measured against an unreliable reference; (2) the elimination of the "wrong output order" family for
> `down` is **withdrawn**, and that family is back in play; (3) only the **residual-axis** oracle
> (+0.700 reference) is sound, and on it `down`'s output axis genuinely does fail (+0.022). The
> negative results for the 48 digit-orderings and the intra-bank `ofast`/`ifast` pair still stand,
> since those were scored on the residual axis too.

Hypotheses tested for `down`, all negative:

| hypothesis | FFN axis | residual axis |
|---|---|---|
| `ofast` (current: `o = slot % 16`, `i = slot // 16`) | +0.005 | −0.010 |
| `ifast` (transposed: `o = slot // 3200`, `i = slot % 3200`) | +0.011 | +0.010 |
| all 48 `(blk,bank,o)` digit orderings + reversals | best +0.057 | -- |
| bit-reversal / z-order shuffles (`bitrev10`, bank-only, o-only) | no better | -- |
| *reference* | **+0.222** | **+0.700** |

Everything sits at noise on the sound (residual-axis) oracle. `down`'s L-class geometry is confirmed
(16 outputs x 3200 inputs per bank; 4 blocks x 16 banks x 16 = 1024 outputs), so the defect lies in how
a bank's 51,200 nibbles map to `(input, output)` pairs and/or how the composite output index is
assembled -- it is **not** the obvious row-major/column-major pair, nor any digit permutation of the
composite index, nor a bit-reversal shuffle.

Note the retraction above: after it, the evidence supports "`down`'s **output** ordering is wrong,
mechanism unidentified", not the stronger "both axes are wrong, therefore it must be the slot
decomposition". The search space is larger than the retracted argument implied, not smaller.

### Consolidated state of the reconstruction

| component | status |
|---|---|
| embedding, tokenizer | correct (tokenizer verified: `The capital of France is` -> `<bos> The(673) capital(5283) of(533) France(7005) is(567)`) |
| hidden RMSNorm gains | correct (gamma = 1, proven positively, `PICO_NORMS.md` addendum) |
| per-head QK-norm | **was missing from the export**; fixed by moving to the `qwen3` architecture |
| `Q`, `K`, `V`, `gate`, `up` | correct (round-trip validated, `OutTrans=0`) |
| `O` | **orientation solved** (stored transposed, two independent confirmations); residual ordering may still be imperfect -- recovers ~48% of reference, not 100% |
| `down` | **unsolved** -- output ordering wrong on the sound oracle; mechanism unidentified (input-axis claim retracted) |

`down` is now the single identified blocker, and it is a harder object than the ordering question it
replaced: the failure of both axes simultaneously means the fix is a change to the slot decomposition,
not a permutation of either axis.
