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

> # ⚠ SECTIONS 20, 21 AND 23 ARE RETRACTED — THE ORACLE THEY USE IS INVALID
>
> Sections 20, 21 and 23 all score candidates with a "residual-basis oracle" resting on one
> assumption that was never checked: that in a correct transformer the magnitude with which a
> **writer** (`O`, `down`) writes residual position *j* correlates with the magnitude with which a
> **reader** (`gate`, `up`, `Q`) reads position *j*. pico's readers agreeing at +0.65 was taken as
> the reference, and the writers scoring ~0 was read as a defect.
>
> `validate_oracle.py` tests that assumption on **Qwen3-4B-Instruct** — a real, correctly-ordered
> model in pico's own architecture family (RMSNorm + per-head QK-norm + GQA + SwiGLU), dequantized
> from Q4_K/Q6_K. Averaged over 12 layers:
>
> | quantity | Qwen3-4B (known correct) | pico (assumed reference) |
> |---|---|---|
> | `O_out ~ gate_in` | **−0.122** | +0.011 |
> | `down_out ~ gate_in` | +0.040 | +0.022 |
> | `gate_in ~ up_in` (the "reference") | **−0.176** | +0.650 |
>
> **The reader-reader reference is itself negative in a correct model.** Writer and reader
> magnitudes do not track each other, so "pico's writers score ~0" is the *expected* result, not a
> defect. The oracle measures nothing about ordering correctness.
>
> **Withdrawn:** the §20 localisation of the defect to the O/down output ordering; the §21 QAP and
> structured-layout verdicts *as evidence about ordering* (the searches ran correctly, but their
> scores are uninformative); the §23 conclusions about `down`. The §19 LoRA Rosetta result and the
> §22 transpose finding are **unaffected** — §19 uses cross-bank rank consistency and §22 was
> independently confirmed by the attention-head signature, neither of which touches this oracle.
>
> **Net effect:** the reconstruction is still broken, but its fault is *not* localised. The claim
> "the defect is the O/down output ordering" is no longer supported by evidence.

## 20. Residual-basis test: the defect is the O/down **output** ordering (a fast static oracle) — RETRACTED, see above

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
| `O` | **stored transposed** -- established by the attention-head signature, which is independent of the retracted oracle |
| `down` | **unknown** -- the oracle that localised a fault here is retracted; no evidence currently isolates the defect |

`down` is now the single identified blocker, and it is a harder object than the ordering question it
replaced: the failure of both axes simultaneously means the fix is a change to the slot decomposition,
not a permutation of either axis.

---

## 24. A control-validated oracle, and what it establishes

Every oracle used before this section was *assumed* to work, and §20's was later shown to be
outright invalid. This section builds one that is **validated against a known-good model first**,
then applies it. All pico numbers come from Apple's shipped asset decoded by our own code; the
reference model is used only as a calibration standard, exactly like checking an instrument
against a known weight. No third-party weights enter pico, the export, or any result.

### The statistic

Derived from the computation rather than from weight statistics. `O` writes residual position *j*
and `gate` reads it, so the composed map contracts over the residual index:

```
residual_j = sum_i attn_i * O[j,i]        ffn_k = sum_j residual_j * gate[k,j]
=> composed:  gate @ O          score = || A @ B ||_F  vs random permutation of the contracted axis
```

Trained networks align successive transformations, so the correct pairing should score higher than
a random re-pairing. (This is the QAP objective with **raw** Gram matrices; earlier attempts used
row-normalised *cosine* matrices, which discard exactly the magnitude information the objective
depends on -- a likely reason they found nothing.)

### Positive control (Qwen3-4B-Instruct, dequantized from Q4_K/Q6_K)

| | identity | random mean | z |
|---|---|---|---|
| 8 layers, `gate @ O` | ~200 | ~160 | **+300 mean** |

The control **passes decisively**: the statistic detects a correct residual pairing at z ≈ +300
against a random one. Only after this was it applied to pico.

### Result 1 -- `O` is stored transposed (now functionally confirmed)

| orientation | z |
|---|---|
| `O` as decoded (`[attn, res]`) | **+2.9** (random) |
| `O` transposed (`[res, attn]`) | **+96 … +105** (strongly aligned) |

This independently reproduces §22's transpose finding with a validated functional statistic, and it
does *not* rely on the retracted magnitude oracle. Together with the attention-head signature, the
transpose is confirmed by three mutually independent methods.

### Result 2 -- the defect is isolated to `down`

Auditing every pairing where two pico tensors share an index (5 layers):

| pairing | best orientation | mean z | |
|---|---|---|---|
| `O -> gate` [residual] | transposed | **+104.7** | aligned |
| `Q -> O` [head] | transposed | **+70.7** | aligned |
| `down -> Q(L+1)` [residual] | as-decoded | +1.4 | random |
| `down -> gate(L+1)` [residual] | as-decoded | +1.3 | random |
| `gate -> down` [ffn] | as-decoded | −2.6 | random |
| `up -> down` [ffn] | as-decoded | −1.4 | random |

**Every pairing involving `down` is at the random level; every pairing not involving it is aligned.**
With O transposed, the rest of the network is correctly decoded. `down` is the sole remaining defect,
and *both* of its axes fail -- the signature of a wrong slot -> (input, output) decomposition inside
the bank rather than a permutation of either axis. (Unlike §20/§23, this isolation rests on a
statistic with a passing positive control.)

### Result 3 -- what `down` is NOT (all scored on the valid oracle)

| hypothesis | best z |
|---|---|
| slot = `(i//G)*(16G) + o*G + (i%G)` for G = 1,2,4,8,16,32,64,3200 | **+2.4** (at G=64) |
| scale applied per-output `sc[o]` (current) / omitted / reversed | +1.1 / +2.1 / +1.0 |
| all 48 `(blk,bank,o)` digit orderings and reversals | previously rejected |
| bit-reversal / z-order shuffles | previously rejected |

Nothing approaches the +105 that the *correct* pairings reach. `down`'s L-class geometry is
confirmed sound (16 outputs x 3200 inputs per bank; 4 blocks x 16 banks x 16 = 1024 outputs), so the
defect is in how a bank's 51,200 nibbles map to `(input, output)` pairs -- and it is none of the
natural candidates.

### Standing state

| component | status |
|---|---|
| embedding, tokenizer, gamma=1, QK-norm | correct |
| `Q`, `K`, `V`, `gate`, `up` | correct |
| `O` | **solved** -- stored transposed, three independent confirmations |
| `down` | **unsolved** -- both axes scrambled; decomposition not recovered |

Six of seven weight roles are now positively confirmed rather than merely assumed. One wrong
residual writer is still enough to destroy the forward, so the model does not yet generate coherent
text -- but the open problem is now a single tensor with a validated test to score candidates against.

## 25. L-class positional read: the decoder is CONFIRMED; the wall is the scale-bearing mode

Rather than continue rejecting candidate decompositions for `down`, the positional read that
originally cracked the N-class was executed at pico's **L-class** geometry
(`Cin=3200 -> Cout=256`, 4-bit palettized, 16 banks). Five probes encode the position digits
(`o0,o1` for `o` in [0,256); `i0,i1,i2` for `i` in [0,3200)), each weight already an integer in
[0,16) so palettization is exact, and each bank decoded through its **own** codebook.

Two practical corrections were needed, both real bugs in the first attempt:
* the coefficient bytes must be located from the parsed `__KERN_0` **file offset** (`0x14000`) and
  the emitted `CoeffSize` (`0x6440`), not by scanning for a "plausible codebook" -- that heuristic
  landed in `__DEBUG` and produced garbage (26 distinct pairs instead of 819,200);
* the header length is `CoeffSize - payload`, derived rather than assumed.

### Result: perfect bijection

```
819,200 distinct (o,i) pairs over 256 x 3200   -- complete, no collisions
o range [0..255]     i range [0..3199]
bank 0 slots  0..15 -> o = 0..15,  i = 0
bank 0 slots 16..31 -> o = 0..15,  i = 1
bank 0 o in [0..15]   bank 1 o in [16..31]

    =>   o = 16*bank + (slot % 16)        i = slot // 16
```

**This is exactly the current decoder.** So the L-class slot decomposition is not the defect: it is
now positively confirmed against ANE ground truth, not merely assumed. That closes the hypothesis
class §24 was searching (chunked slot decompositions), and explains why every candidate there failed
-- the decoder was already right.

### The remaining difference, and the wall

The probe emits `CoeffSize 0x6440` = 64-byte header + 25,600-byte payload. pico's shipped `down`
tiles are `0x6480` = **128-byte header**, the extra 64 bytes being 16 fp16 **scales** + 32 unknown.
The payload size is identical; what is unverified is whether the scale-bearing mode lays that
payload out the same way.

That mode could not be produced with the available toolchain:

| palettization config | emitted |
|---|---|
| `per_grouped_channel, group_size=16, channel_axis=0` | `0x6440` (no scale table) |
| `per_grouped_channel, group_size=1, channel_axis=0` | `0x0680` (per-channel LUT, different layout) |
| `per_tensor + enable_per_channel_scale` | **compile fails** -- `InvalidMILProgram` |
| `per_grouped_channel gs=1 + enable_per_channel_scale` | **compile fails** -- `InvalidMILProgram` |

Both scale-bearing configurations lower to `constexpr_blockwise_shift_scale`, which this machine's
ANE compiler rejects outright, before any layout is emitted. This is the same wall recorded in §5
(candidate 2), now confirmed against the modern `mlprogram -> coremlcompiler -> mil_to_hwx` path
rather than the legacy one.

### Where this leaves `down`

Positively established: the L-class geometry, the codec, and the slot decomposition are all correct.
Established by the control-validated oracle (§24): `down` nonetheless fails every pairing, on both
axes, while every non-`down` pairing is aligned.

Those two facts together mean the defect is **not** in the plain-mode layout the probe can reach. It
is in what the scale-bearing mode does differently -- most likely how the 16 per-bank scales attach
to the payload -- and that mode is currently **unreproducible on this toolchain**, so it cannot be
read positionally. This is a tooling limit, stated as such: not "we could not find the permutation",
but "the one mode that differs cannot be compiled here to be measured".

## 26. The `0x6480` mode is a BIAS, not a scale table -- and the L-class decode is confirmed in pico's exact mode

§25 hit a wall: pico ships `CoeffSize 0x6480` (128-byte header) but every scale-bearing
palettization config failed to compile (`InvalidMILProgram`), so the shipped mode could not be
positionally read. That wall is now broken, by a route that avoids the rejected op entirely.

### What actually produces the 128-byte header

Testing fusion patterns that use only ops the compiler accepts:

| graph | emitted |
|---|---|
| `conv` (palettized) | `0x6440` |
| `conv -> mul` by per-output-channel constant | `0x6440` |
| **`conv` with a `bias`** | **`0x6480`** |
| `conv` with `bias`, then `mul` | `0x6480` |

The extra 64 bytes appear when the conv carries a **bias** -- not from per-channel scaling. Verified
directly by compiling with `bias[o] = 100 + o` and reading the header back:

```
bank 0  header[64:96] as fp16 : 100 101 102 ... 115     (= bias for outputs  0..15)
bank 1  header[64:96] as fp16 : 116 117 118 ... 131     (= bias for outputs 16..31)
```

Exact match. So that header slot carries a **per-output-channel bias** in this mode.

**This does not mean pico's decoder is wrong.** pico's values in the same slot are all *positive* and
tightly banded (0.073–0.443 for `O`, 0.082–0.443 for `down`, 0.106–0.155 for `gate`), while its
codebook spans ~±0.8. That is the signature of a per-output **scale** (codebook x scale gives
plausible weight magnitudes), not a bias -- LLaMA-style models carry no biases, and a uniformly
positive bias on every output would be unexplainable. The slot is evidently reused by mode. The
decoder's `cb[nb] * sc[o]` is therefore retained.

### Positional read in pico's exact shipped mode

With `bias` forcing `0x6480`, the L-class positional read was rerun in the mode pico actually ships:

```
819,200 distinct (o,i) pairs over 256 x 3200  -- PERFECT BIJECTION
o = 16*bank + (slot % 16)        i = slot // 16      (identical to the 0x6440 result)
```

**The payload layout does not change between the two modes.** The L-class decode -- geometry, codec,
and slot decomposition -- is now confirmed against ANE ground truth *in pico's own configuration*.

### The audit is sound: per-pairing controls on a known-good model

`down`'s pairings involve an attention block or the SwiGLU nonlinearity, so they are not equivalent to
the direct `O -> gate` pairing and needed their own controls. On Qwen3-4B (correct):

| pairing type | Qwen3 (correct) | pico |
|---|---|---|
| `O -> gate` [direct] | +406 | **+105** (transposed) |
| `up -> down` [SwiGLU between] | +407 | −1.4 |
| `down -> gate(L+1)` [attention between] | +328 | +1.3 |
| `gate -> down` [SwiGLU between] | +57 | −2.6 |

Every pairing type scores strongly in a correct model, including across attention and SwiGLU. So the
pairings are informative and **`down` is genuinely broken** -- the §24 audit stands.

### `down`: what has now been eliminated, on the validated oracle

| hypothesis | best z | control |
|---|---|---|
| slot decomposition | **confirmed correct** by positional read (bijection) | -- |
| 4-block output assembly, all 24 permutations | +1.1 | +328 |
| composite `(blk,bank,o)` digit orderings + reversals (48) | +2.5 | +328 |
| intra-bank `ofast` / `ifast`, chunked `G = 1..3200` | +2.4 | +328 |
| scale applied / omitted / reversed | +2.1 | +328 |
| QAP on **raw** Gram matrices, group level | +3.3, cross-layer agreement 2.6% (chance 1.6%) | +328 |

Every level of the decode that can be tested is confirmed correct, and every reordering hypothesis is
rejected against a control that reaches +328. `down` remains broken, and the cause is now *not*
locatable within the decode model as currently understood -- which points at an assumption upstream
of it (for example whether the four L-blocks the weight map assigns to `down` are in fact that
tensor) rather than at any ordering within it.

## 27. Block-to-role assignment audited; `s`-block position tested

Dumping layer 0's 40 blocks in file order shows a clean, unambiguous role layout, so the weight
map's assignment is not obviously wrong:

```
Q    4 x N        K  1 x N       V  1 x N       O  4 x N
gate 1 x s + 12 x N              up 1 x s + 12 x N
down 4 x L
```

with a uniform `+0x20800` stride between N blocks (`+0x10800` after each `s`, `+0x64800` between
`L`s) -- i.e. the blocks are contiguous and correctly classed, and there is no spare or misfiled
block inside the layer.

One genuine discrepancy surfaced: the `s` block is stored **first** for `gate`/`up`, while
`src/pico_forward.py` documents `s_position = "append_cols"` (the `s` half-block occupying the
**last** 128 output columns). Since `gate`'s output axis is what `gate -> down` contracts, a wrong
`s` position there would break that pairing without `down` being at fault. Tested both assemblies on
the validated oracle:

| `s` position | mean z (`gate -> down`, 4 layers) |
|---|---|
| first (current) | −0.8 … +1.6 |
| last (`append_cols`) | +0.2 … +1.8 |
| *Qwen3 control for this pairing type* | **+57** |

Neither is right. So the `s` position is not the cause either, and the block-to-role assignment is
sound as far as this audit can reach.

### Standing conclusion

Every component of the `down` decode that can be checked has now been checked and is correct --
geometry, codec, slot decomposition (positional read, bijection, in pico's own `0x6480` mode), block
classing, and block-to-role assignment. Every reordering hypothesis has been eliminated against a
control that reaches +328. Yet `down` fails every functional pairing while the other six roles pass.

That combination is not explained by anything currently in the decode model. The remaining
possibilities are all *outside* it: the four L-blocks may be a tensor other than `down` despite the
consistent layout, the `OutTrans=1` mode may alter the payload in a way the bias-forced probe does
not reproduce (the probe reaches `0x6480` but still compiles at `OutTrans=0`), or the shipped tiles
carry a transform with no counterpart in anything compilable here. Distinguishing these needs a
weight-bearing `OutTrans=1` conv, which remains unreachable on this toolchain (§18).

## 28. LoRA cross-reference attempts: one weak signal, one invalid test (recorded, not relied on)

Two further attempts used Apple's shipped `lora_32_constant_data.bin` as an independent handle on
`down`, since those tensors are unpalettized fp16 in the confirmed `OutTrans=0` mode and are DMA'd
verbatim (§19), so they carry no codec ambiguity.

**Attempt 1 -- the LoRA delta as a stand-in for `down`.** `dW = A_down @ B_down` is `[3200, 1024]`,
the same shape and index conventions as base `down`, so it can be scored on the validated oracle
without touching the base tile decode:

| | mean z |
|---|---|
| `gate -> dW` (LoRA) | +0.6 |
| `gate -> base down` | −0.3 |
| `dW -> gate(L+1)` | −0.0 |
| `base down -> gate(L+1)` | +0.1 |
| *Qwen3 controls* | **+57 / +328** |

The LoRA delta fails too. Read narrowly this hints that `gate`'s **output** axis may be implicated
rather than `down` alone -- but a rank-32 delta is a weak probe of a full-rank composition, so this
is recorded as a hint, not a finding.

**Attempt 2 -- diagonal dominance of `W^T dW`: INVALID, discarded.** The idea was that if a base
tile and its adapter share index conventions, `W^T @ dW` should be diagonally dominant. Measured, it
is at chance everywhere (argmax-on-diagonal 0.0000–0.0020 against a 1/n chance of ~0.0003–0.001;
diag/off-diag ratio 0.99–1.03) for `gate`, `up` **and** `down` alike.

That null is **uninformative and is not used as evidence**, because the premise is wrong: `dW = A @ B`
is a low-rank update, and `W^T dW = W^T A B` has no reason to be diagonal -- diagonal dominance would
require `dW ~ c*W`, which is not how LoRA trains. Unlike every other test in §24-§27 there is also no
way to control it, since the reference model ships no adapters. Recorded here so the negative is not
mistaken later for evidence that `gate`/`up`/`down` are all mis-ordered.

## 29. `down`'s VALUES are correct; the fault is ordering on both axes -- and both are independently confirmed

A test that separates two things earlier sections conflated. **Permutation preserves singular
values**, so the spectrum distinguishes "wrong order" from "wrong values":

| role | s1/s10 | s1/s100 | eff. rank |
|---|---|---|---|
| `Q` | 2.11 | 4.11 | 546 |
| `O` | 2.41 | 5.07 | 502 |
| `gate` / `up` | 2.81 / 2.98 | 5.33 / 4.53 | 834 / 880 |
| **`down`** | **2.11** | **2.90** | 806 |
| *random control* | **1.02** | **1.14** | 940 |

`down` is unambiguously **trained**, nothing like the random control. So its decoded *values* are
right and the defect is purely **ordering** -- which also means the codec, scales and slot map are
all working, consistent with the positional read.

The two oracle tests are also cleanly **independent**, which was not previously noted: permuting
`down`'s columns cannot change `||gate @ down||_F`, and permuting its rows cannot change
`||down @ gate(L+1)||_F`. So `gate -> down` probes *only* down's input axis and
`down -> gate(L+1)` probes *only* its output axis. Both fail, so **both axes are independently
misaligned** -- neither result is an artefact of the other.

### A structural asymmetry that explains why

`gate`'s residual axis is its **`i` axis** (`i = slot // 16`), and `O`'s residual axis is *also* its
`i` axis (that is what the §22 transpose amounts to) -- which is exactly why `O -> gate` aligns at
+105. But `down`'s residual axis is its **`(blk, bank, o)` composite**, and its FFN axis is the `i`
axis while `gate`'s FFN axis is *its* composite. So on both of `down`'s axes a *composite* index is
being matched against an *i* index. Two different constructions of the same physical channel order.

That is a coherent explanation for both failures at once. What it does not yield is the mapping
between them: every transformation tried is rejected against a +328 control.

| hypothesis (this round) | gate->down | down->gate(L+1) |
|---|---|---|
| identity (current) | +0.1 | +1.2 |
| rolls of the 3200 axis (+-128, +-256), reverse | best +2.5 | -- |
| gate-composite order / reversed blocks | +1.1 | -- |
| swap `bank`<->`o` within 256-chunks (both axes) | −0.9 | −0.7 |
| reverse `o` / reverse `bank` within chunks | −0.7 | −1.1 |
| scale indexed by `i//200`, `(i//16)%16`, `i%16` | best +1.1 | best −0.2 |
| *Qwen3 control* | **+57** | **+328** |

Note the scale variants also **lower** `s1/s100` (3.25 current -> 2.6-2.8), i.e. they make the
spectrum flatter/less trained -- independent evidence that the current per-output scale indexing is
the right one and should not be changed.

### Assessment

Everything checkable about `down` is confirmed correct, its values are confirmed trained, the failure
mode is understood structurally (composite-vs-`i` index construction), and every concrete mapping
between the two constructions has been rejected against a strong control. Further blind enumeration
is not a good use of effort: the space of "plausible reorderings" has been covered, and the answer is
evidently not in it.

The one variable never isolated remains `OutTrans=1` itself. Every tile that decodes correctly is
`OutTrans=0`; both tiles needing intervention (`O`, `down`) are `OutTrans=1`; and `O`'s fix turned out
to be a **transpose**, not a permutation -- a class of transform no reordering search can reach. The
bias trick reaches pico's `0x6480` *header* but still compiles at `OutTrans=0`, so the payload effect
of that flag has never been observed. Obtaining a weight-bearing `OutTrans=1` conv requires a
toolchain that compiles it, which this host does not provide.

## 30. Per-role residual-axis census, and a rejected linear-assignment method

### Which axis of each role is the residual?

Scoring every role against `gate` (whose residual axis is its `i` axis) on the validated oracle,
testing both orientations:

| role | col-test (composite is residual) | row-test (`i` axis is residual) | convention expects |
|---|---|---|---|
| `Q` | +2.0 | **+300.7** | row (input = res) |
| `K` | n/a | **+184.9** | row |
| `V` | n/a | **+174.5** | row |
| `up` | n/a | **+523.9** | row |
| **`O`** | +6.3 | **+160.1** | col (output = res) -- **violated** |
| **`down`** | **+0.6** | n/a | col (output = res) |

`Q`, `K`, `V`, `up` all behave exactly as the convention predicts, at very high z. `O` does **not**:
its `i` axis is the residual, which is the storage transpose of §22 seen a third independent way.
`down`'s composite -- which the convention says *is* its residual output -- fails at +0.6.

So the canonical residual ordering is the `i`-axis ordering shared by `Q/K/V/up/O`, and `down`'s
composite ordering does not match it. `down` cannot be "stored transposed" the way `O` is: its
L-class block structure fixes `i` at 3200 and the composite at 1024, so the assignment is forced.

### Rejected: linear assignment on FFN-space affinity

`down[:,c]` and `gate[j,:]` both live in FFN space (R^3200), so a direct channel-to-channel affinity
`A[j,c] = |gate[j,:] . down[:,c]|` exists and matching them is a **linear** assignment -- exactly
solvable by Hungarian, unlike the NP-hard quadratic assignment every earlier attempt reduced to.
That looked like a genuine methodological upgrade.

**The control kills it.** Run on Qwen3-4B, where the true answer is the identity:

| | recovered identity fraction | chance |
|---|---|---|
| L0 / L1 / L2 | 0.0008 / 0.0004 / 0.0016 | 0.00039 |

The affinity carries **no ordering information even in a correct model**. pico's apparent gains under
the same procedure (z +5.5, +0.8, **+31.7**, +2.0) are therefore overfitting, and the recovered
permutations agree across layers **0.0%** of the time (chance 0.10%). Recorded so the +31.7 is not
mistaken later for a partial solve.

### Status

The failure is now characterised as precisely as static analysis allows: `down`'s values are correct
(trained spectrum), its geometry/codec/slot map are proven against ANE ground truth, its composite
residual ordering demonstrably differs from the canonical `i`-axis ordering that all six other roles
share, and no transformation between the two has survived a control. Every method that produced a
positive-looking pico number has been controlled, and each one that could be controlled failed the
control.

## 31. The FFN axis has no external anchor -- which is why it cannot be pinned

A structural observation that explains the shape of every failure in §23-§30.

The 3200-wide FFN channel axis is referenced by **only four tensors**: `gate` (output), `up`
(output), `down` (input), and the LoRA `A_down` (input). There is no fifth, independently-verified
tensor touching it. The residual axis, by contrast, is touched by `Q`, `K`, `V`, `O`, `gate`, `up`
and `down`, which is exactly why the residual ordering could be established (§30: `Q` +300.7,
`up` +523.9, `O` +160.1 all agreeing).

Every pairing available over the FFN axis fails:

| pairing | pico | Qwen3 control |
|---|---|---|
| `gate -> down` | +0.2 | +57 |
| `up -> down` | −0.6 | +407 |
| `gate -> A_down` (LoRA factor, `ofast` / `ifast`) | +1.0 / +2.2 | -- |
| `B_down -> gate` (LoRA factor, residual axis) | +0.4 | -- |

The LoRA **factors** were tested separately from their product here for the first time, precisely
because the product is a weak probe. They fail too. So no available tensor pair fixes the FFN
ordering, and there is nothing outside this set to appeal to.

### `gate`'s 13-block output assembly: also tested, also negative

§27 tested `down`'s 4-block assembly but never `gate`'s 13-block one (1 x `s` + 12 x `N`), which
defines the FFN axis. Sixteen assemblies (s-first, s-last, N reversed, and all 11 rotations of the
12 `N` blocks), scored on the validated oracle:

| best candidates | `gate -> down` | `up -> down` |
|---|---|---|
| `s` last | +2.1 | +1.9 |
| `s` last, N reversed | +1.6 | +2.2 |
| identity (current) | +0.2 | −0.6 |
| *Qwen3 control* | **+57** | **+407** |

Nothing approaches the control.

### Decode sanity confirmed

The decoded `down` tensors are genuine and distinct, ruling out a map or offset error:

```
L0..L3  mean|W| 0.0258-0.0263   std 0.0331-0.0363   zeros 0.0000   finite 1.0000
corr(L0,L1) 0.0155   corr(L0,L2) 0.0167   corr(L1,L2) 0.0139     (distinct layers, as expected)
block offsets distinct and evenly spaced across layers
```

### Where this leaves the static approach

`down`'s values are correct, its geometry/codec/slot map are proven against ANE ground truth, its
tensors are well-formed and distinct, and the residual axis is firmly established from six agreeing
roles. What cannot be established from the shipped data alone is the **FFN channel ordering**, because
every tensor that could anchor it is itself one of the tensors in question -- the constraint system is
underdetermined, not merely unsolved. That is a structural reason for the failures, not a run of bad
luck, and it means further enumeration of orderings is not expected to succeed.

Breaking it needs an anchor from outside the weight data: a weight-bearing `OutTrans=1` compile (to
read the layout directly), or a runtime capture of the FFN activation tensor. Neither is available on
this host.

## 32. Xcode A/B: the wall is the OS, not the toolchain (corrects earlier guidance)

Earlier sections suggested that obtaining "a different Xcode" might emit the weight-bearing
`OutTrans=1` conv that everything hinges on. **That guidance was wrong**, and this section corrects
it with a direct test.

Both toolchains present on this host were driven through the same graphs:

```
Xcode 26.5      (17F42)      /Volumes/D/Xcode.app
Xcode 27.0 beta (27A5218g)   /Applications/Xcode-beta.app     (matches the OS)
```

| graph | Xcode 26.5 | Xcode 27.0 beta |
|---|---|---|
| full real MHA | 12 tasks, `OutTrans=1` on 2, **weight-bearing: 0** | identical |
| L-class in the `0x6480` bias mode | 1 task, `OutTrans=1`: 0 | identical |
| L-class + `enable_per_channel_scale` | **fails** -- `InvalidMILProgram` | **fails** -- identical |

The two toolchains are **byte-for-byte equivalent in outcome**, including the same failure.

### Why

`mil_to_hwx` links against:

```
/System/Library/PrivateFrameworks/ANECompiler.framework      <- SYSTEM framework
/System/Library/PrivateFrameworks/ANEServices.framework
```

`xcrun coremlcompiler` (which *is* Xcode-provided) only produces the `.mlmodelc`; the actual ANE
compilation -- the stage that assigns `OutTrans` and rejects `constexpr_blockwise_shift_scale` -- is
performed by **ANECompiler.framework, which ships with macOS, not Xcode**. Changing Xcode changes the
MIL front end only, never the ANE backend. Hence the identical results.

Every other `ANECompiler.framework` on disk (in both Xcodes' SDKs and both CommandLineTools SDKs) is
a `.tbd` **text stub for linking**, not an implementation. No simulator runtimes are installed. So
this host has exactly **one** ANE compiler, tied to macOS 27.0 build 26A5388g.

### Corrected requirement

The unlock is **a different macOS version** (hence a different `ANECompiler.framework`), not a
different Xcode. Concretely, either:

* another Mac running a different macOS build, where the same probes are re-run; or
* an installed simulator runtime / older SDK that ships a real ANECompiler binary rather than a
  `.tbd`, loaded ahead of the system one via `DYLD_FRAMEWORK_PATH`.

Neither is available here, and no amount of Xcode switching substitutes for it.

### Simulator runtimes cannot supply an alternative ANECompiler

§32 proposed loading a different `ANECompiler` from a simulator runtime via `DYLD_FRAMEWORK_PATH`,
as the only route not needing second hardware. **That route is closed, for a structural reason.**

Three iOS runtimes are installed (18.4 / 26.5 / 27.0, plus two watchOS). Checked directly:

| | |
|---|---|
| PrivateFrameworks inside the iOS 18.4 runtime | **1,979** (so the image is fully populated) |
| `ANE*.framework` in that runtime | **0** |
| `ANECompiler` across all three iOS runtimes | **0 / 0 / 0** |
| `Vision.framework` espresso weight files named `*_nonane` | **12** |

The `_nonane` suffix is Apple's own marker for "no ANE". The iOS Simulator executes on the host
Mac's CPU/GPU and has **no Neural Engine**, so no ANE compiler or ANE services ship in any runtime
image. This is by design, not a missing download -- no simulator runtime of any version will ever
provide one.

**Consequence:** every on-machine route to a different ANE compiler is now eliminated. The system
`ANECompiler.framework` (macOS 27.0, build 26A5388g) is the only one that exists here, and the
requirement reduces to a Mac running a different macOS build.

## 33. What `OutTrans=1` actually DOES: a strided, interleaved output write (from Apple's L2 registers)

A source never previously mined: each ANE task descriptor carries **L2 buffer addressing** --
`L2_Src1Base`, `L2_ResultBase`, and per-axis strides `L2_*Strides: C=...`. These describe where in
ANE local memory a task reads and writes its channels, i.e. the **activation layout**, straight from
Apple's binary.

### The flag is perfectly determined by the result stride

Over all **988** weight-bearing tasks in pico's `binary_0.hwx`:

| `OutTrans` | `SrcStrideC` | `ResStrideC` | count | example |
|---|---|---|---|---|
| 0 | `0x90` | **`0x80`** | 675 | 1024 -> 256 |
| 0 | `0x80` | **`0x80`** | 133 | 1024 -> 256 |
| 1 | `0x80` | **`0x810`** | 144 | 1024 -> 256 |
| 1 | `0x80` | **`0x800`** | 36 | 32 -> 1024 |

> **`OutTrans=1` tasks with a large result stride: 180 of 180.
> `OutTrans=0` tasks with a large result stride: 0 of 808.**

A perfect, exceptionless correlation. `OutTrans=0` writes channels **contiguously** (`0x80` = 128 B =
64 fp16, exactly one row). `OutTrans=1` writes them **16x strided** (`0x800` = 16 x 128 B, plus a
`0x10` pad in the 1024->256 case).

### The down-projection's actual addresses

```
task 202  3200 -> 256   Src 0x0e8000 (strideC 0x80)   Res 0x14c040   strideC 0x810   OutTrans=1
task 203  3200 -> 256   Src 0x0e8000 (strideC 0x80)   Res 0x14c240   strideC 0x810   OutTrans=1
task 204  3200 -> 256   Src 0x0e8000 (strideC 0x80)   Res 0x14c440   strideC 0x810   OutTrans=1
task 205  3200 -> 256   Src 0x0e8000 (strideC 0x80)   Res 0x14c640   strideC 0x810   OutTrans=1
```

Two facts follow directly:

* **`down` reads its 3200 FFN inputs CONTIGUOUSLY** (`SrcStrideC = 0x80` from the SwiGLU buffer at
  `0x0e8000`, which the preceding elementwise task 200 writes in place). So down's *input* axis is
  the natural contiguous FFN order -- it is not permuted at the activation level.
* **`down`'s four tasks write INTERLEAVED**: bases `0x14c040/240/440/640` are `0x200` apart while the
  per-channel stride is `0x810`. Since `0x200` = 4 channels and `0x810` ~ 16 channels, the four tasks
  each contribute 4 channels into every 16-channel block.

This is the first direct evidence of what `OutTrans=1` does, and it confirms the flag is an
**activation-layout** property -- consistent with §19 (the LoRA Rosetta showed the *coefficient*
order is unchanged) and with §25/§26 (the positional read gave the same bijection in both header
modes).

### The derived mapping -- tested, only marginally better

Reading the interleave as `physical = (j//4)*16 + t*4 + (j%4)` (four tasks x four channels per
16-block) and applying it to `down`'s output:

| | mean z, `down -> gate(L+1)` |
|---|---|
| current decode | +0.5 |
| L2-derived interleave | **+1.4** |
| *Qwen3 control* | **+328** |

Directionally right but nowhere near correct, so the exact byte arithmetic is still off: `0x810`
is `16 x 0x80 + 0x10`, and that `0x10` padding means channel slots are not uniformly spaced, so the
naive division is not the true index map. The ANE register maps shipped with the tooling document
`PlaneStride` fields but do not define the `OutTrans` layout, and the second `OutTrans` field in
`PECfg` is not emitted for these task types.

### Why this matters anyway

The flag has gone from an unexplained scheduler decision to a **measured, quantified layout
transform** whose parameters are visible in the binary. The remaining unknown is narrow and concrete:
how the `0x810` stride with its `0x10` padding maps task-local channel `j` to a physical residual
channel. That is a question about ANE L2 tiling arithmetic, and it can be answered exactly by a
weight-bearing `OutTrans=1` compile on any host that emits one -- which is precisely what the probe
kit tests for.

## 34. Cross-machine and cross-configuration result: the compiler is not the variable

§32 concluded that a different macOS build was the remaining requirement, since `ANECompiler`
ships with the OS. That has now been tested on real hardware, and the conclusion must be tightened.

### Two macOS builds, a MAJOR ANECompiler version apart -- identical output

| | reference host | second machine |
|---|---|---|
| macOS | 27.0 (26A5388g) | **26.5.2 (25F84)** |
| `ANECompiler` | 10.24.3 | **9.509.0** |
| Xcode | full 26.5 / 27.0b | CommandLineTools only |
| compiled via | `xcrun` | `coremltools` fallback |

| graph | tasks | `OutTrans=1` | weight-bearing |
|---|---|---|---|
| `bare_conv` | 1 | 0 | 0 |
| `Lclass_bias` | 1 | 0 | 0 |
| `transpose_conv_add` | 3 | 1 | **0** |
| `real_MHA` | 12 | 2 | **0** |
| `real_MHA_x2` | 24 | 4 | **0** |

**Byte-identical on both machines.** A 10.x -> 9.x gap is a major-version difference, not a point
release, and it changed nothing. (The first attempt on that machine produced a *false* negative: it
had only CommandLineTools, so `xcrun coremlcompiler` was missing and nothing compiled. The kit now
falls back to `coremltools` + the system CoreML framework and distinguishes an environment failure
from a real result -- both runs above are real, with 5 of 5 graphs compiled.)

### ANE architecture targets and optimisation flags -- also null

| target | `OutTrans=1` | weight-bearing | with `-O` (no-optimize) |
|---|---|---|---|
| h13 | compile fails | -- | fails |
| h14, h15 | **0** | 0 | fails (code 22) |
| h16, h17, h18 | 2 | **0** | fails (code 22) |

`OutTrans` is a newer-generation feature -- h14/h15 never emit it at all, h16+ emit it only on
weightless shuffle tasks. pico's shipped hwx has the same CPU type/subtype (`0x0080` / `0x0007`) as
the probe builds, so the target arch was already correct.

### What this changes

The search space is now: **2 macOS builds x 2 ANECompiler major versions x 2 Xcode versions x 6 ANE
arch targets x 2 optimisation settings x 5 graph shapes** -- and every compilable combination gives
the same answer. `OutTrans=1` never lands on a conv that carries coefficients.

So the earlier framing ("find a Mac with a different ANECompiler") is **substantially weakened**. The
compiler version is demonstrably not the variable. Apple's shipped model has 180 weight-bearing
`OutTrans=1` tasks, so their build pipeline produces something none of these configurations does --
most plausibly an internal toolchain or a graph construct that cannot be expressed through
`coremltools`' MIL front end. Trying additional macOS versions is now low-value; the evidence says
the difference is upstream of the compiler, in how the graph reaches it.

## 35. RETRACTION: the LoRA slicing is not validated, so LoRA-derived evidence is unreliable

`lora_32_constant_data.bin` was used as ground truth in §19 (the "Rosetta stone"), §28, and again
when sweeping output maps for the weight-bearing `OutTrans=1` LoRA up-projection. Those uses all
assume the per-layer role offsets inside that file are correct. **That assumption fails validation.**

### The validation

If the adapter belongs to these base weights, then `gate_A` -- shape `[1024 residual, 32]` -- has a
residual axis that must match `gate`'s own residual input axis, which is independently verified
(§30: `up -> gate` row-test **+523.9**). Contracting them:

| | mean z over 4 layers |
|---|---|
| `gate_A^T @ gate` | **−0.4** |
| `up_A^T @ up` | **−1.1** |
| `O_A^T @ gate` | **+1.2** |
| *correct residual contraction, base tensors* | **+160 … +523** |

All at noise. The adapter's residual axis does not correspond to the base weights' residual axis for
any role tested.

### What is and is not affected

**Still solid:** the segment discovery itself. `__MKERN_0`/`__MKERN_9` have `File Size 0` and VM sizes
of exactly 29,687,808 / 59,375,616 bytes, matching `lora_32/64_constant_data.bin` byte-for-byte, and
the file is layer-major with period 618,496 (autocorrelation peak 0.893 at exactly that lag, and again
at 2x). Those facts stand.

**Now unreliable:** the interpretation of the file's *internal* layout. The role-offset assignment was
inferred from log-RMS boundaries, and it never fully closed -- the K/V region produced two 16,384-element
segments that match no rank-32 role (expected sizes are only 32768, 8192 and 102400). That unresolved
discrepancy is now the likely explanation for every LoRA result being at noise.

**Consequently withdrawn:**
* §19's *evidence* that `OutTrans=1` leaves the intra-bank coefficient order unchanged. **The
  conclusion survives** -- it is independently established by the positional read (§25/§26), which
  used synthetic probes and no LoRA data, and found the identical bijection in both the `0x6440` and
  the shipped `0x6480` header modes. Only the LoRA route to it is retracted.
* §28's LoRA-delta comparison (already reported as a weak hint, now void).
* This section's own sweep of output maps for the `32 -> 1024` `OutTrans=1` LoRA tensor: best z +1.1,
  which cannot be read as a negative result about `OutTrans` layouts, because the input slicing is
  unvalidated.

### Lesson recorded

The LoRA file looked like ideal ground truth -- unpalettized fp16, DMA'd verbatim, spanning both
`OutTrans` modes -- and that made it tempting to trust without checking it against something already
known. The cheap validation (does its residual axis match a verified residual axis?) should have been
run before any conclusion was drawn from it, not after three sections had relied on it.

## 36. LoRA segmentation SOLVED exactly -- but it still cannot be validated

§35 retracted the LoRA offsets as unvalidated. They have now been re-derived properly.

### Method that worked

The earlier attempt took boundaries from the largest log-RMS *jumps*, giving 15 segments including
two of 16,384 elements -- a size matching no rank-32 role. The fix is to segment by RMS **level**
rather than by jump: LoRA `A` and `B` are initialised differently (`B` starts at zero), so the
profile is a clean two-level signal that run-length encodes directly. The recovered run lengths are
then required to be a permutation of the architecturally-fixed multiset -- a hard check the old
approach lacked.

Required multiset (rank 32, D=1024, FFN=3200, KV=256): `32768 x9, 8192 x2, 102400 x3` = 618,496.

**Layers 0 and 1 match it exactly, 14 runs each** (layer 2 shows +-256 jitter from the median
filter, trivially snapped). Perfect `HIGH/low` alternation confirms `A`/`B` pairing.

### The recovered layout

| A offset | len | B offset | len | role, from the size signature |
|---|---|---|---|---|
| 0 | 32768 | 32768 | 32768 | Q or O |
| 65536 | 32768 | 98304 | 32768 | Q or O |
| 131072 | 32768 | 163840 | 8192 | K or V |
| 172032 | 32768 | 204800 | 8192 | K or V |
| 212992 | 32768 | 245760 | 102400 | gate or up |
| 348160 | 32768 | 380928 | 102400 | gate or up |
| **483328** | **102400** | **585728** | **32768** | **down** (unique signature) |

**The role order is `Q/O, K/V, gate/up, down` -- not the `Q,K,V,O,gate,up,down` previously assumed.**
Consequences for the earlier work: `down_A`/`down_B` (483328 / 585728) and `gate_A` (212992) were
**already correct**; `O_A` (147456) was **wrong** -- no segment begins there, so it straddled two.

### Why this still does not unlock anything

With `gate_A`'s offset now *confirmed* correct, its residual axis still fails to align with `gate`'s
verified residual axis, under every unpack convention tried (`ofast` −1.7…+4.3, `ifast` −0.9…+0.7,
input-banked −1.3…+1.0, against +160…+523 for correct base-tensor contractions).

The reason is a limitation of the test, not necessarily of the data: **there is no controllable check
for whether a LoRA factor should align with its base weight at all.** `A` is randomly initialised and
trained, so its per-channel directions need not track `W`'s, and the reference model ships no adapters
to calibrate against. This is the same failure mode as the diagonal-dominance test in §28 -- an
uncontrollable statistic whose null result carries no information.

### Net

The segmentation is a **real and reusable result**: exact, self-consistent, checked against a
constraint that could have falsified it, and it corrects the role order. But it does not rehabilitate
the LoRA file as ground truth for the `OutTrans` question, because no available test can confirm the
correspondence. The §35 retraction therefore stands in substance -- LoRA-derived conclusions about
coefficient layout remain unusable -- while the offsets themselves are now known.

## 37. Channel layout read directly from Apple's L2 addresses -- directionally right, still short

§33 established that the task descriptors carry the activation layout. Tracing a full layer's
`L2_ResultBase` values yields the actual channel assignment, which had until now been assumed.

### gate/up: the file order is REVERSED, and `s` is last

The 13 tasks that build one FFN tensor write to descending addresses as the task index rises:

```
task 181  Cout=128  Res 0x0c4000      <- the 's' block
task 182  Cout=256  Res 0x0bc000
...                       (-0x8000 each, = 256 channels x 128 B)
task 193  Cout=256  Res 0x064000
```

The tensor spans `0x064000 .. 0x0c8000` = 409,600 B = 3200 x 128 B exactly. Converting address to
channel index (`(addr - 0x064000) / 0x80`):

* task 193 (the **last** file block) -> channels **0..255**
* task 182 -> channels 2816..3071
* task 181, the `s` block (**first** in the file) -> channels **3072..3199**, i.e. **last**

So the weight map's file order is **reversed** relative to logical channel order, and the `s`
half-block sits at the **end**. Both facts are read from Apple's binary, not inferred.

### Measured effect

| | mean z |
|---|---|
| `gate -> down`, file order (current decoder) | +0.3 |
| `gate -> down`, **L2-derived** gate map | **+2.3** |
| `up -> down`, L2-derived | +2.0 |
| *Qwen3 control* | +57 |

Then, holding the L2-derived gate map fixed and sweeping `down`'s output map:

| `down` output map | mean z |
|---|---|
| contiguous `t*256+j` (current) | +0.4 |
| task-interleaved `t+4j` | −1.1 |
| **L2-derived `(j//4)*16 + t*4 + (j%4)`** | **+3.0** |
| reversed task `(3-t)*256+j` | +0.7 |
| *Qwen3 control* | +328 |

### Reading this honestly

Every L2-derived map improves its baseline, consistently and across all layers tested
(+0.3 -> +2.3 for gate, +0.4 -> +3.0 for down). That is a real signal and the direction is almost
certainly right -- these maps come from Apple's own addressing rather than from a guess.

But the magnitude is an order of magnitude short of a correct pairing. The likely reason is
arithmetic that these traces do not pin down: the `OutTrans=1` stride is `0x810 = 16 x 0x80 + 0x10`,
so channel slots inside a block are **not uniformly spaced**, and any clean formula like
`(j//4)*16 + t*4 + (j%4)` is at best an approximation of the true tiling. Resolving the remaining
padding arithmetic needs either the ANE tiling specification or a weight-bearing `OutTrans=1` compile
to read it off -- and §34 showed no available compiler configuration produces one.

The decoder default is left unchanged: a partial map that scores +3 against a +328 reference is not
evidence enough to ship, and adopting it would silently bake in an unverified layout.

## 38. What `OutTrans=1` really is: a sequence/channel transpose -- retracting the "interleave" reading

§33 and §37 read the `OutTrans=1` result stride (`0x810`) as evidence that the four down-projection
tasks *interleave their output channels*, and derived the map
`physical = (j//4)*16 + t*4 + (j%4)` from it. **That reading is wrong and is retracted.** Tracing the
consumer of that buffer settles the semantics exactly.

### The decisive observation

The down-projection tasks write the buffer at `0x14c000`; the task that consumes it reads the *same*
memory with a different shape:

```
producer (tasks 202-205)   W=64    Cout=256   Res 0x14c000 + t*0x200   ResStrideC 0x810
consumer (task 3217)       W=1024  Cin=64     Src 0x14c000             SrcStrideC 0x810
```

`W` and `C` are exchanged. The buffer spans `0x14c000`--`0x16c400` = 132,096 B = **64 rows x 2064 B**,
and `2064 = 1024` fp16 `+ 16` B padding -- an exact fit for the consumer's view, and
`4 x 256 x 64 = 64 x 1024 = 65,536` elements either way.

### The address arithmetic closes

Each task writes, in every one of the 64 rows, 256 consecutive fp16 at row-offset `t*256`
(`t*0x200` bytes = `t*256` fp16). So producer element `(task t, channel j, sequence w)` lands at

```
0x14c000 + w*2064 + t*512 + j*2
```

which is consumer element `(row = w, position = t*256 + j)`.

\begin{quote}
**Therefore the residual channel is `p = t*256 + j` -- plainly contiguous -- and
`ResultStrideC = 0x810` is the stride over the *transposed* channel axis (i.e. the original
*sequence* axis), not over output channels.**
\end{quote}

### Consequences

* `OutTrans=1` is a **sequence/channel transpose of the activation**. It does **not** permute output
  channels. This is fully consistent with the positional read (§25/§26), which found the identical
  coefficient bijection in both header modes, and with the flag being an activation-layout property
  (§33's headline conclusion survives; only its channel-interleave corollary does not).
* The §33/§37 interleave map, and the +3.0 score attributed to it, are **withdrawn**. That score was
  an artefact of scrambling channels in a way that happened to score marginally above the identity,
  not evidence of a recovered layout.
* **`down`'s output channel ordering is contiguous `t*256 + j` -- exactly what the decoder already
  does.** So the "wrong output ordering" hypothesis for `down` is now closed from the hardware side,
  not merely unsupported.

That is a real narrowing: of the two axes §23 implicated, the output axis is now positively accounted
for. It also deepens the puzzle, because the composition oracle still reports
`down -> gate(L+1)` at the random level while every non-`down` pairing is aligned. Since the output
map is now known-correct and the input side reads the FFN buffer contiguously (§33), the residual
discrepancy has to lie in how the FFN buffer's 3200 channels are assembled by `gate`/`up` -- the one
axis §31 showed to be underdetermined, and for which §37's L2-derived reversal (`s` last, file order
reversed) improved `gate -> down` from +0.3 to only +2.3 against a +57 control.

## 39. The full FFN dataflow, and an unresolved contradiction

Tracing every task that touches the down-projection's output completes the dataflow and confirms
§38 independently -- while leaving a contradiction that this investigation has not resolved.

### The dataflow

```
gate tasks 168-180   ->  0x000000..0x064000   (contiguous, strideC 0x80; s task FIRST -> channels 3072..3199)
up   tasks 181-193   ->  0x064000..0x0c8000   (same pattern)
SwiGLU (EW, in place)->  0x0e8000             (3200 channels, contiguous strideC 0x80)
down tasks 202-205   ->  0x14c000             (OutTrans=1, transposed view, strideC 0x810)
...
task 213             ->  0x040000             (OutTrans=1: Src strideC 0x800, Res strideC 0x90)
next layer 214-217   <-  0x040000             (Q/K/V/O read the residual, SrcStrideC 0x90)
```

Task 213 is the **transpose-back**: it reads the strided `[W=1024, C=64]` view and writes the
contiguous residual layout (`0x90` = 64 fp16 + 16 B pad), which the next layer's projections then
read. So the residual channel equals the position index `p` in the transposed buffer, and by §38
`p = t*256 + j`.

**`down`'s output ordering is contiguous, confirmed twice from independent address traces.** It is
what the decoder already does.

### Both `gate`/`up` channel maps are also now read from hardware

Both FFN tensors follow the same pattern, and the `s` half-block behaves identically in each:

| task | class | result base | channels |
|---|---|---|---|
| 168 / 181 | `s` (Cout=128) | `0x060000` / `0x0c4000` | **3072--3199** (last) |
| 169 / 182 | `N` | `0x058000` / `0x0bc000` | 2816--3071 |
| ... | | (−`0x8000` per task) | |
| 180 / 193 | `N` | `0x000000` / `0x064000` | 0--255 |

Each tensor spans exactly `0x64000` = 409,600 B = 3200 x 128 B. The `s` block is **first in task
order and first in file order, but holds the last 128 channels**, and the `N` blocks run backwards.

### The contradiction

Applying that map lifts `gate -> down` only from +0.3 to +2.3 (control +57), and the residual-side
pairing is worse:

* `down`'s output map is **proven contiguous** (two independent traces).
* `gate(L{+}1)`'s input axis is **independently verified** as the residual (§30: `up -> gate`
  row-test +523.9).
* `||\,\mathrm{down}\cdot\mathrm{gate}(L{+}1)\,||_F` is invariant to permuting `down`'s *rows*, so
  only these two -- both verified -- determine it.
* Yet it measures **+1.3**, at the random level, against a +328 control.

Every premise in that chain has been checked, and they cannot all be true. The most likely
explanation is that the composition oracle, though calibrated on a correctly-ordered reference model
and sound for the pairings that *do* align (`O -> gate` +104.7, `Q -> O` +70.7), fails for `down`
specifically for a reason not yet identified -- rather than that the hardware traces are wrong.

That is where this line of work stops: not at an untested hypothesis, but at a conflict between
measurements that are each individually well-supported. Recording it as such is more useful than
selecting whichever premise would let a story close.

### Addendum: the unexamined header bytes are not a value modifier

The L-class coefficient header is 128 B: codebook `[0:32]`, zeros `[32:64]`, per-output scales
`[64:96]`, and `[96:128]` which earlier notes label "unknown". If that region held a second scale or
a bias, the decoded *values* would be wrong independently of any ordering question, which would
explain the contradiction above. It does not.

Read across banks, `[96:128]` is high-entropy and structureless -- e.g.\ `0x98544b66 0x56787861
0x4ca99783 ...`, whose fp16 interpretation gives wild mixed-magnitude values
(`14.8, -0.002, 35872, 103.5, ...`, and `nan` in places). It shows none of the regularity of the
scale region immediately before it (uniformly positive, tightly banded 0.08--0.44). It is far more
consistent with a checksum or DMA descriptor than with numeric weight metadata, and the header/payload
split is independently fixed by arithmetic: `CoeffSize 0x6480 = 25728 = 128 + 25600`, and the payload
must be exactly 25600 B to hold `16 x 3200` nibbles.

So the decoded values are not being modified by an overlooked header field, and the contradiction in
§39 stands unexplained.

## 40. External corroboration: the published ANE reference validates the decode

Until now every conclusion here came from first-principles analysis of the shipped binary. A
reverse-engineered ANE reference has since been published (arXiv:2606.22283, *Apple Neural Engine:
Architecture, Programming, and Performance*, ~300 pp., web edition at `ane-guide.readthedocs.io`),
which documents the compiler, on-disk program format, weight-compression scheme and memory
hierarchy. Checking this work against it resolves several open questions -- all in favour of the
decode as implemented.

### 1. `OutTrans` is a fused transpose epilogue (confirms §38)

The reference lists the epilogue slots one engine layer can absorb, in order:

```
ZinTextureLayer     in-place spatial/texture remap
ZinBroadcastLayer   broadcast of a fused operand
ZinActivationLayer  pre-GOC activation
ZinGOCLayer         the gain/offset (scale + bias) unit
ZinActivationLayer  post-GOC activation
ZinTransposeLayer   output transpose / layout      <-- OutTrans
ZinQuantLayer       output (re)quantization
```

and names "transpose and reshape chains" among the fusable epilogues. So `OutTrans=1` marks a
convolution that **absorbed a following transpose into its output stage**. This is exactly the
conclusion of §38, reached independently from the L2 traces, and it confirms the corollary: the flag
changes the *output layout only* and leaves the coefficient stream untouched -- consistent with the
positional read finding the same bijection in both header modes (§25/§26).

Note also the slot order: the GOC (scale/bias) is applied **before** the transpose, so the per-output
scale indexes the convolution's own output channel, not a transposed one. The decoder's `sc[o]`
indexing is therefore correct.

### 2. The `0x10` padding is bank-conflict avoidance, not layout

> "The pool is interleaved across 64 banks at a 16-byte granule, with the bank index
> `floor(addr / 16) mod 64`, and a compile-time stride optimizer spreads accesses to avoid conflicts."

This settles the arithmetic §37 could not pin down. The `OutTrans=1` stride `0x810` is
`0x800 + 0x10` = one logical row of 1024 fp16 **plus a single 16-byte granule**, inserted by the
stride optimizer to shift the bank index and avoid conflicts. It is **not** part of the logical
channel layout, so the mapping is the plain contiguous one derived in §38 -- and the "non-uniform
slot spacing" worry recorded in §37 was unfounded.

### 3. The 128-byte coefficient header is the four fused-conv streams

The reference documents that a fused convolution emits four kernel-coefficient streams:

| stream | reloc register | role |
|---|---|---|
| bias | `0x1554` | per-channel additive offset |
| post-scale | `0x1558` | per-channel output multiply, **where a dequantize scale folds** |
| palette lookup | `0x155c` | the palettized-weight lookup table |
| activation lookup | `0x1560` | 33-segment piecewise-linear activation table |

That maps directly onto the observed header: `[0:32]` the 16-entry palette (**palette lookup**),
`[32:64]` all zeros (**bias**, absent in this model), `[64:96]` uniformly positive banded values
(**post-scale**, the folded dequantize scale). This independently validates the codec as
implemented, and confirms §26's reading that the shipped values in that slot are scales rather than
biases even though a *bias* is what causes the header to be emitted at `0x6480`.

Because a linear map satisfies `(W·x)·s = ((W·s)·x)` per output channel, folding the post-scale into
the weights -- what the decoder does -- is mathematically identical to applying it as an epilogue.

### 4. `OCG` matches the observed bank structure

Output channels are tiled into output-channel groups sized to the accumulator file
(`OCG = min(floor_pow2(8/(kW·kH·kD)), byte_cap)`), with `OCG passes = ceil(Cout/OCG)`. pico's tasks
report `OCGSize=4`; a group of 16 output channels is consistent with the positional read's finding
that **16 output channels vary fastest** within a bank, and with 16 scales per bank.

### 5. One fact not previously accounted for

> "The engine stores tensors **channel-interleaved**, and a channel dimension that is not a multiple
> of the interleave factor is padded out to one... The width axis aligns to a 16-byte
> direct-memory-access granule."

Tensor order is `[N, D, C, H, W]`. The interleave factor is family-specific, read from the
hardware-abstraction table, and is not given as a literal. This concerns *activation* storage rather
than the coefficient stream, and pico's `W=64` fp16 (128 B) is already a multiple of the 16-byte
granule, so no silent padding applies -- but the channel-interleaved organisation is the one
documented layout property this work has not independently verified.

### What this changes

Every element of the decode that the reference covers is confirmed correct: codec, header semantics,
scale handling, `OutTrans` meaning, group size, and the padding that had blocked §37. That
substantially strengthens the reading of the §39 contradiction: since the decode matches the
documented format at every checkable point, the discrepancy is more likely in the **composition
oracle's applicability to `down`** than in the recovered weights.

Sources: arXiv:2606.22283; `ane-guide.readthedocs.io`; `github.com/freedomtan/coreml_to_ane_hwx`;
`github.com/skyfallsin/apple-neural-engine-field-guide`.

## 41. Functional test of the hardware-derived corrections: negative

§40 confirmed the decode against the published ANE reference and concluded the composition oracle was
the likelier culprit. The correct response is to stop scoring with a suspect statistic and test
**functionally** -- actual generation, which is ground truth. Four builds, differing only in the two
corrections at issue:

| build | `O` | `gate`/`up` channel map | output |
|---|---|---|---|
| `qwen3` (baseline) | as decoded | file order | word-salad |
| `qwen3-Ot` | **transposed** | file order | **empty** |
| `hw` | **transposed** | **L2-derived** | **empty** |
| `hw2` | as decoded | **L2-derived** | word-salad |

All four load cleanly, tokenize correctly (`<bos> The(673) capital(5283) of(533) France(7005)
is(567)`), and hold sane tensors (finite, no zeros, `mean|w|` 0.022--0.031). These are genuine
forward-pass results, not environment failures.

**Neither hardware-derived correction produces coherent text.**

* The **L2-derived `gate`/`up` map** (`s` block last, `N` blocks reversed) changes nothing visible.
  It is read directly from Apple's task addressing and is almost certainly *right*, but it is
  evidently not the operative defect.
* The **`O` transpose** makes generation *degenerate* -- the model emits nothing even with
  `--ignore-eos`, in both builds that apply it. This is evidence against it functionally, and §40
  supplies a reason it may have been a misreading: the reference documents `OutTrans` as
  `ZinTransposeLayer`, an epilogue that transposes the **output activation at runtime**. That does
  not imply the stored coefficient matrix is in `[out, in]` order. The §22/§30 statistical evidence
  for a transpose is real but was gathered with instruments that have misled before, and the
  functional test does not support it.

### Standing

The reconstruction now has: a codec confirmed against the published format, a slot decomposition
confirmed by positional read in the shipped header mode, channel maps read from Apple's own L2
addressing, and correct tokenization, norms and architecture. It still does not generate coherent
text, and the corrections that hardware evidence most strongly supports do not change that.

That is a narrower and better-evidenced failure than before, but it is a failure, and the honest
reading is that at least one assumption still standing is wrong in a way none of the available
instruments -- statistical or hardware-derived -- has isolated.

## 42. An NLL oracle, an invariance that retracts sec.41, and a third RMSNorm

Sections 39-41 left the composition oracle discredited and the hardware-derived corrections
functionally useless. This section replaces the instrument and re-derives what is actually unknown.

### 42.1 The forward harness is faithful

A NumPy forward reading the GGUF directly reproduces llama.cpp's word-salad continuation for the same
prompt. Hypotheses can therefore be tested without rebuilding a GGUF each time.

### 42.2 Teacher-forced NLL, with a known-good control

Generated text cannot discriminate anything while one tensor is wrong -- every variant is salad. NLL
is continuous and has a hard reference scale: uniform over the vocabulary is `ln(262144) = 12.477`.
The same ablations were run on Qwen3-4B, whose weights are known good:

| | chance | intact | attn-only (`ffn_down=0`) | ffn-only (`attn_output=0`) |
|---|---|---|---|---|
| Qwen3-4B (control) | 11.93 | **1.47** | 18.14 | 11.44 |
| pico (this work) | 12.48 | **13.33** | 14.66 | 13.31 |

Two things follow, and the second corrects an inference I had been about to make.

1. pico shows **no language-modelling signal at all** -- it is at or slightly above chance.
2. Deleting the FFN from a *perfectly good* model costs 16.7 nats and lands **far worse than
   chance**. So pico sitting at chance is entirely consistent with "every role correct except
   `down`". Being at chance is **not** evidence of a second defect, and an earlier attempt to read
   it that way -- via an attention-only/FFN-only text ablation -- was uninformative for the same
   reason: the control, ablated identically, is also incoherent.

### 42.3 Joint-permutation invariance: sec.41's gate/up remap could never have mattered

Applying the L2-derived channel map to gate/up **and the same permutation to down's input axis**
returns NLL `13.325` -- bit-for-bit the base value. This is not a coincidence but an identity: the
FFN is invariant under any joint permutation of gate/up's output channels and down's input channels,
because the two index the same 3200 neurons. Applying it to one end only (as the `hw` build did)
scores `16.514`.

**Consequences.** The L2-derived channel map is *functionally irrelevant*; sec.41's `hw`/`hw2` builds
tested nothing about it, and its "read directly from Apple's task addressing, almost certainly right"
framing was beside the point. The only meaningful unknown in the FFN is the **relative** permutation
between down's input axis and gate/up's output axis -- one degree of freedom, not two.

### 42.4 The relative permutation: swept, and null

67 structured bijections of 3200 (every factorization transpose, block reversals, rotations, and
transpose-then-reverse) were scored by NLL against random-permutation controls, then the top
candidates re-tested with 6x the tokens and 24 controls:

| hypothesis | NLL | controls beating it |
|---|---|---|
| `T(4,800)+revblk` | 12.545 | 1/24 |
| `revblk(80)` | 13.048 | 2/24 |
| `identity` (current decode) | 13.164 | 3/24 |
| **best random control** | **12.476** | -- |

The best of 24 *random* permutations beats every structured hypothesis, including the sweep's winner
and including identity. **No candidate is distinguishable from noise.** In particular the exploratory
round's apparent "identity beats shuffled" signal (13.33 vs 13.87) did not survive more tokens and
more controls -- it was small-sample noise, and the claim that the current ordering is "partially
right" is withdrawn.

### 42.5 A third RMSNorm per layer

The shipped graph contains **142 `ANE_RMSNorm` against 45 `ANE_ScaledDotProductAttention`** -- a ratio
of 3.16 -- with `ANE_QKNorm` (46) a *separate* op family. Every GGUF built here uses `qwen3`
architecture, which has exactly **two** norms per layer. So the graph carries one normalization per
layer that the reconstruction does not.

Since gamma=1 is already established for pico, the extra norm is parameter-free and costs nothing to
test. All four placements were tried in the NumPy forward -- pre-norm only (baseline), post-attention,
post-FFN, and both (Gemma2-style sandwich). **All four are incoherent**, and NLL separates none of
them from chance.

Two readings remain open, and the evidence here does not choose between them:

* the third norm is real and sits somewhere not yet tried (on the residual stream, on the KV path, or
  on the embedding), or
* it belongs to the **LoRA/adapter path** rather than the base model. The graph is the *adapted*
  model -- it ships `lora_32`/`lora_64` constant data -- and `ANE_MultiOutputLinear` shows the same
  ~2.9-per-block ratio, which is what an extra adapter projection per block would look like. On this
  reading the third norm is not part of the base model at all and correctly has no place in the GGUF.

### Standing

The instrument problem is now fixed: NLL with a known-good control model and random-permutation
controls is a real oracle, unlike the composition z-score. Applied honestly it says the FFN's
relative channel permutation is **one** unknown rather than two, that none of 67 structured
candidates for it survives contact with controls, and that the current decode's ordering is not
measurably better than random. That is a sharper statement of the blocker than sec.41 could make,
and it is still a blocker.

## 43. The embedding is proven correct, the QK-norm was self-inflicted, and V/O are broken

Section 42 established a controlled NLL oracle and found the FFN's relative permutation
underdetermined. Pushing further with that oracle produced three results that change the picture,
and one retraction.

### 43.1 The embedding and token mapping are correct (first ironclad component proof)

At depth 0 the model predicts the **current** token at 600/600 positions -- correct tied-embedding
behaviour. That test is *invariant to any permutation of embedding rows*, so on its own it proves
nothing about the token-id to row mapping; my own notes flag exactly this as vacuous. The semantic
geometry settles it:

```
 Paris  ->  Paris, PARIS, Paris(fr), Bei-jing-glyph form, paris, Parisian, Parizh(ru)
 dog    ->  dogs, Dog, DOG, Dogs
 three  ->  Three, two, four, trois, drei, tres
```

Multilingual semantic neighbourhoods in every case. **The embedding and the token-id mapping are
correct.** This is the first component of the reconstruction validated positively and
unambiguously, and it makes the embedding usable as ground truth for testing other tensors.

### 43.2 The QK-norm was self-inflicted, and it was destroying attention at evaluation time

Attention entropy was measured against the uniform bound `ln(t)`:

| config | mean sharp fraction | max over layers |
|---|---|---|
| RoPE half, **QK-norm on** | 0.0012 | 0.0038 |
| RoPE half, **QK-norm off** | **0.1809** | **0.7073** |

With the per-head QK-norm applied, attention is uniform at all 24 layers -- no head is sharp
anywhere. Remove it and real structure appears, rising with depth to 0.71 at layer 19. All gammas
are exactly 1.0, so this is not a bad gamma: applying a per-head RMS normalization over 64 dims is
itself wrong for this model.

This matters beyond attention. The `qwen3` architecture was adopted specifically to add QK-norm,
which means **every FFN search ever run here -- the 67-hypothesis sweep of sec.42, and the ~15
hypothesis classes before it -- was scored through a forward whose attention had been flattened to
noise.** No FFN hypothesis could have shown a signal under those conditions.

Related nulls, all measured: RoPE `half` (LLaMA/NeoX) beats interleaved (GPT-J) and no-RoPE on
attention structure, so the convention is right. `head_dim = 64` (16Q/4KV) beats 32, 128 and 256 --
so the `QKNorm[128]` note of task 11 does **not** imply 128-wide heads. Embedding scale helps at no
value tested; `sqrt(D) = 32` makes NLL worse, and the residual trace explains why (RMSNorm makes
every sublayer scale-invariant, so the scale only shifts the copy/predict balance).

### 43.3 Re-running the FFN sweep with attention working

With the QK-norm removed, five of 67 hypotheses beat all 20 random controls, where previously the
best *random* permutation beat every hypothesis:

| hypothesis | NLL | controls beating it |
|---|---|---|
| `T(4,800)+revblk` | 12.582 | 0/20 |
| `transpose(40,80)` | 12.709 | 0/20 |
| `revblk(64)` | 13.032 | 0/20 |
| `identity` | 13.323 | 1/20 |
| best random control | 13.320 | -- |

Held to the same standard as before, this is **still not significant**: with 67 hypotheses against 20
controls, ~3.2 are expected to beat all controls under the null, and 5 were observed.
`T(4,800)+revblk` has now won three independent runs on different token sets and different attention
settings, which is worth recording but is not proof. A joint sweep over embedding scale x
permutation found no cell below chance, and `top1-next` is **0/299 in all 25 cells**.

### 43.4 V and O are broken -- a second blocker, with an FFN-independent test for it

Because the embedding is now trusted, the OV circuit can be tested on its own. If attention selects
one token, a head contributes `rms(e_s) @ Wv.T @ R @ Wo.T` with `R` the GQA repetition map; this is
independent of the attention pattern and of the entire FFN. Scored as
`cos(rms(e) @ A_L, e)` against a shuffled-token control:

| model | mean abs z | max abs z |
|---|---|---|
| Qwen3-4B (known good) | -- | **8.2** (2.0-4.5 in early layers) |
| pico, as decoded | **0.04** | 0.10 |
| pico, **O transposed** | **0.39** | 0.54 |

The metric is real -- it registers strongly on a working model. pico registers essentially nothing.
**V and/or O are mis-decoded**, which is a second blocker and was not previously known; it also means
the "six of seven roles proven" claim -- which rested on the composition oracle retracted in sec.39 --
does not survive. Transposing `O` improves the metric tenfold and consistently across all layers,
which agrees with the independent statistical evidence of sec.22/30, but at 0.39 against a control of
2-8 it is clearly not the whole story. GQA head order (blocked vs interleaved) makes no difference.

### Standing

The blocker is no longer "one tensor". Confirmed correct: the embedding, the token mapping, the
tokenizer, the RoPE convention, the head configuration, and gamma=1. Confirmed broken: V/O, by a
metric validated on a known-good model. Undetermined: the FFN's relative channel permutation, whose
entire search history was conducted through a forward crippled by a QK-norm this work added itself.

The useful development is methodological. The OV test needs no working forward and no FFN, so V/O
can now be attacked directly instead of end-to-end -- which is the first time any single tensor role
in this reconstruction has had an isolating test with a positive control.

## 44. V/O: values are trained, layout is not recoverable by any coarse permutation

Section 43 introduced the OV-circuit test and showed pico registers ~0 where a known-good model
registers strongly. This section establishes how sensitive that test is, and how far the V/O defect
goes.

### 44.1 The metric is layout-sensitive -- verified, not assumed

Scrambling O's input axis on Qwen3-4B collapses the score by two orders of magnitude:

| Qwen3-4B layer | identity | random permutation (max of 10) |
|---|---|---|
| 0 | 3.583 | 0.066 |
| 2 | 3.777 | 0.069 |
| 5 | 4.587 | 0.062 |
| 34 | **8.576** | 0.074 |

So the test genuinely measures layout, and a model with correct V/O cannot hide from it. pico scores
**0.386 with identity and 0.387 for the best of 12 random permutations** -- it is at its own noise
floor. The 0.386 is therefore *not* evidence of partial correctness, and sec.43's tenfold
"improvement" from transposing O must be read only as an orientation effect, not as recovered
structure. (Note the orientation asymmetry is still real and still agrees with sec.22/30: the
dimensionally-valid convention for the control is `A = Wv.T R Wo.T`, whereas pico only scores
above floor as `A = Wv.T R Wo` -- i.e. pico's O is stored transposed relative to the GGUF
convention.)

### 44.2 The values are trained weights

Stable rank `||W||_F^2 / ||W||_2^2` and the ratio of largest to median singular value, against a
matched Gaussian and against Qwen3-4B:

| model | tensor | stable rank | s0/s_med |
|---|---|---|---|
| random | Gaussian 1024x1024 | 257.6 | 2.47 |
| pico | attn_v | 27.3 | 3.65 |
| pico | attn_output | 16.2 | 15.89 |
| pico | ffn_down | 47.5 | 5.84 |
| Qwen3-4B | attn_output | 63.6 | 7.59 |
| Qwen3-4B | ffn_down | 155.5 | 4.36 |

Every pico tensor is heavy-tailed and far from random. **The decode recovers genuine trained
matrices**; what is wrong is where the numbers sit, not what they are.

### 44.3 No coarse layout hypothesis recovers V/O

Swept against the metric, all with random-permutation controls:

* orientation: `Wo` vs `Wo.T`
* 27 structured bijections of O's input axis (1024) and 21 of V's output axis (256) -- every
  factorization transpose, block reversal, and transpose-then-reverse
* **joint** head-layout conventions, which a per-axis sweep cannot express because V's output axis
  and O's input axis must agree: head-major (`h*64+d`) vs dim-major (`d*nh+h`) on each, times three
  KV-to-Q head assignments (blocked, interleaved, blocked-reversed) -- 24 combinations

Every joint configuration scored **0.362-0.386**, flat against a 0.39 noise floor. The dim-major
convention recorded in the pico head-layout note does not help, alone or jointly.

### Standing

Confirmed correct: embedding, token mapping, tokenizer, RoPE convention, head count, gamma=1, and now
the *values* of every decoded tensor. Confirmed broken: the placement of V/O, by a metric with a
positive control that is 100x sensitive to exactly this. Ruled out for V/O: orientation alone, all
coarse axis permutations, and all head-layout conventions.

What remains for V/O is therefore the same class of problem already known for the down projection --
a fine-grained **intra-tile** scramble, where the ANE tile decode emits correct values at wrong
positions below the granularity any coarse permutation can express. That is a considerably more
specific statement of the defect than "V/O are wrong", and unlike the FFN's channel permutation it
comes with a fast isolating test that needs no working forward.

## 45. Exact permutation solving, and what the attention block actually contains

Brute force over O's input axis is not available -- 1024! arrangements. But the copy objective
factorizes, which allows the exact optimum to be computed instead of searched.

### 45.1 The layout problem is a linear assignment problem

With `H` the rms-normalized embedding sample, `En` the L2-normalized one, and `G = H^T En`:

```
sum_t (h_t A) . e_t  =  <A, G>_F
A = B M[pi],  B = Wv.T R        =>   <A,G> = sum_a Y[a,:] . M[pi[a],:],   Y = B^T G
                                =>   score(pi) = sum_a Cost[a, pi[a]],    Cost = Y M^T
```

so the best permutation over all `1024!` is the Hungarian solution of a 1024x1024 cost matrix --
seconds, and globally optimal rather than sampled.

**The solver was validated before being trusted.** On Qwen3-4B layer 0, a known permutation was
planted and the pipeline asked to invert it:

| | z |
|---|---|
| intact | 3.692 |
| scrambled by `pi_true` | 0.010 |
| **Hungarian solution** | **2.473** |

It recovers two thirds of the signal (exact positional agreement is only 245/4096, because many
coordinates are near-degenerate -- the score is what matters). The machinery works.

Applied to pico it returns **nothing**: 0.279 on the very layers it was fit to, against 0.355 for the
identity and a random band reaching 0.42. A validated global optimizer finding no improvement on its
own training layers is strong evidence that **the V/O defect is not a permutation of O's input axis
at all.**

### 45.2 Tile-level rearrangements: also null

The ANE stores these as 128x128 tiles, and two rearrangements lie outside the row-permutation group:
per-tile transpose `(tr,r,tc,c)->(tr,c,tc,r)` and tile-grid transpose `(tr,r,tc,c)->(tc,r,tr,c)`,
swept on V and O independently and jointly, times orientation. Best configuration: **0.388**, against
a 0.39 noise floor. Null.

### 45.3 What the attention block actually contains

A null on the OV metric has two readings -- V/O is broken, or pico simply has no OV-copy structure
and the metric does not apply to it. Sections 43-44 assumed the first. This control decides it, and
simultaneously closes the open question of whether Q/K were ever verified (they were not; sharp
attention had been taken as evidence, but random matrices can also produce sharp attention).

Layer-0 attention computed directly from embeddings, so the degraded residual stream plays no part:

| | sharp fraction | H/H_uniform |
|---|---|---|
| pico, **real** Q/K | **0.0468** | 0.796 |
| pico, **random** Q/K (norm-matched Gaussian) | 0.0050 | 0.860 |
| Qwen3-4B, real Q/K | **0.4149** | 0.518 |

Two conclusions. First, pico's Q/K carry **genuine signal -- 9x above norm-matched random** -- so the
decode is not noise, and pico does exhibit structure where structure exists. That is what makes the
V/O null attributable: the metrics are applicable to this model. Second, Q/K are also **~9x below a
correct model**, so they are *partially* decoded, not correct.

### Standing, corrected

The attention block as a whole is partially decoded, which is a broader statement than sec.43-44 made:

* **Q/K** -- real but degraded signal (9x random, 9x below control). Previously described as
  "confirmed" on the strength of sharp attention; that was never a controlled claim and is now
  measured properly.
* **V/O** -- at the noise floor under orientation, the exact optimum over all 1024! row permutations,
  all head-layout conventions, and all tile transposes.

Both are consistent with a fine-grained intra-tile scramble that leaves some positions correct --
which is exactly what "9x above random, 9x below correct" looks like for Q/K, and what a total loss
of the OV pairing looks like for V/O.

## 46. A real decode bug, found and proven: the tile payload starts at +96, not +128

Sections 43-45 established that the attention block is only partially decoded and that no coarse
layout hypothesis repairs it. Looking for the cause rather than more hypotheses turned up an actual
bug in the decoder.

### 46.1 The tell: four hot channels shared by every tensor

Per-input-channel magnitude profiles were compared across tensors that share the residual basis. In
Qwen3-4B these are essentially uncorrelated (-0.03 to 0.20). In pico:

| pair | pearson | spearman |
|---|---|---|
| L0.Q(in) vs L0.gate(in) | **0.907** | 0.146 |
| L0.Q(in) vs L5.Q(in) | **0.899** | 0.148 |
| L0.Q(in) vs L0.O(**out**) | 0.075 | 0.046 |

High Pearson with near-zero Spearman means a few extreme channels shared by every decode -- an
artifact, not trained structure. They are channels **1020-1023**, the last four of the input axis,
at 3.2x the median norm. Four input channels is exactly 64 nibbles = **32 bytes**.

### 46.2 The tile header is 96 bytes, and the proof is the padding

Dumping a tile header:

```
+0..32      codebook, 16 fp16, cleanly sorted   (-0.791 -0.604 -0.461 -0.360 ...)
+32..64     zeros (reserved)
+64..96     scales, 16 fp16, small positives    ( 0.089  0.123  0.114  0.140 ...)
+96..128    FULL ENTROPY -- 27-32 distinct bytes of 32; as fp16 it is nan / 3.8e4
```

The decoder read the payload from **+128**, treating `+96..128` as header. The decisive test is the
other end of the tile: if the payload runs `96 -> 8288`, the final 32 bytes must be padding.

**All 64 tiles have bytes `8288..8320` exactly zero.** So the layout is

```
header 96 | payload 8192 | zero pad 32   =  8320 = 0x2080 stride
```

and the same 96-byte header with 64-byte alignment reproduces the other two classes exactly:
`96 + 4096 -> 4224 = 0x1080` ('s') and `96 + 25600 -> 25728 = 0x6480` ('L'). The stride alone never
disambiguated this -- `128 + payload` also matches all three -- which is why it survived so long.

### 46.3 Effect

| payload offset | Q/K sharpness | ch1020-23 / median |
|---|---|---|
| 120 | 0.0446 | 2.64 |
| **128 (previous)** | 0.0468 | 3.20 |
| **96 (correct)** | **0.6058** | **1.02** |

A 13x improvement in attention sharpness, sharply peaked at 96 alone, and the outlier artifact
disappears. Re-sweeping the remaining decode assumptions on the corrected payload confirms the
`omin` z-order decisively (0.47-0.61 versus 0.008-0.02 for the transposed rule).

**Every decode-variant experiment in this project ran against a payload shifted by 32 bytes** -- the
same class of error as the QK-norm in sec.43, and their negative results are void.

### 46.4 What it does not fix

Honesty about scope. With the corrected offset:

* OV circuit z remains **0.003-0.10** across all variants -- V/O is still at the noise floor.
* Teacher-forced NLL is **13.372**, still above chance (12.477); `top1-next` is still 0.
* Generation is still incoherent.

Also worth flagging rather than celebrating: sharpness 0.6058 *exceeds* Qwen3-4B's 0.4149, and the
raw attention logits reach 33-53. That may indicate saturation rather than superior structure, so
the sharpness figure should not be read as "better than a real model".

### Standing

One genuine, structurally proven decode bug is fixed -- the first defect in this project identified
by its cause rather than inferred from a metric. It substantially improves Q/K and removes an
artifact that contaminated every tensor. It does not by itself make the model work, so at least one
further defect remains, and V/O is still the outstanding one.

## 47. After the +96 fix: what changed, what did not, and the first sub-chance result

Section 46 proved the payload offset bug. Since that voids every earlier decode-variant result,
the decisive experiments were repeated on the corrected weights.

### 47.1 Still null after the fix

* **V/O layout.** Orientation and head order: 0.027-0.042, unchanged at the noise floor. The
  Hungarian solve now shows textbook overfitting -- train layers lift to 0.099-0.224 while held-out
  layers stay at 0.011-0.060. No transfer, so no real permutation was found. V/O survives the fix
  as the outstanding defect.
* **Embedding scale.** Best value 16.0 at NLL 13.019, still above chance; `sqrt(D) = 32` gives
  13.196. Null, as before but now on trustworthy weights.
* **Norm placement.** pre 13.372, postattn 13.430, postffn 13.395, sandwich 13.515. The plain
  pre-norm arrangement remains best, so the graph's third RMSNorm still has no working placement --
  consistent with it belonging to the LoRA path.

### 47.2 The first configuration to beat chance

Per-layer teacher-forced NLL on the corrected decode:

```
L0    L1    L2    L3    L4    L5    L6    L7    L8    L9   L10   L11
12.5  12.7  12.7  12.4  11.7  11.9  12.0  12.1  12.0  12.2  12.2  12.1
L12   L13   L14   L15   L16   L17   L18   L19   L20   L21   L22   L23
12.3  12.5  12.9  13.0  13.0  13.1  13.3  13.3  13.4  13.4  13.5  13.4
```

Chance is 12.477. **Layers 3-13 are below it, bottoming at 11.728 at layer 4.** This is the first
time any configuration in this project has produced a genuine sub-chance result, and it is a
structural statement rather than a lucky score: a *truncated* model works, and the error accumulates
monotonically with depth. Early layers therefore carry real signal, and each layer injects damage
that eventually swamps it -- exactly what a broken V/O in every block would do.

### 47.3 The layer-0 magnitude anomaly

The residual trace still shows layer 0's attention output at **33.65x** the residual norm
(`||x||` 1.19 -> 64.16, copy fraction 1.000 -> 0.000). `||o||` is 33-40 at *every* layer, so the
embedding at 1.19 is the outlier. `sqrt(1024) = 32` would match it exactly, which is why the Gemma
scaling hypothesis is so tempting -- but it is measured as null (13.196 vs 13.372), because RMSNorm
makes each sublayer scale-invariant and the scale only shifts the copy/predict balance.

### Standing

The +96 fix is real and proven, and it moved Q/K sharpness 0.0468 -> 0.6058 and produced the first
sub-chance NLL. It did not fix V/O, and the model still does not generate coherent text. The
remaining defect is now sharply characterised: something in every block that is *cumulative* --
harmless for three or four layers, fatal by twenty-four -- with V/O the only role still measured at
its noise floor.

## 48. Continuing past the +96 fix: four more hypotheses, all null, and a reasoning correction

### 48.1 The residual basis (O's output axis)

The cumulative signature -- harmless for 3-4 layers, fatal by 24 -- is what a misaligned residual
basis produces: Q/K/V/gate/up *read* the residual, O and down *write* it, and a permutation between
the two mixes the stream a little more each block while every tensor still looks internally fine.
The earlier Hungarian run permuted O's *input* axis; the residual is O's *output* axis, and it has
its own exact solver (`Cost = Y^T O` with `Y = (V R)^T G`).

Solved on five layers, validated on seven held out. **Null, and worse than before**: the solution
does not improve even the training layers (0.056/0.042/0.015/0.048/0.003 against identity's
0.010/0.033/0.078/0.009/0.033). There is not enough signal in the objective to overfit, let alone
transfer.

### 48.2 Scale indexing

An N bank is 16 outputs x 1024 inputs with 16 fp16 scales, which is equally consistent with
per-output scaling and with group-wise input quantization at 64 inputs per group -- the more common
4-bit scheme, and mis-indexing it would damage every tile slightly.

| scale index | L0 sharpness | NLL |
|---|---|---|
| **per-output (current)** | **0.5966** | 13.422 |
| per-input-group (64) | 0.4777 | 13.711 |
| input mod 16 | 0.4390 | 13.100 |
| none | 1.0000 (saturated) | 13.545 |

Per-output is confirmed. The `none` row is instructive rather than competitive: without scales
attention saturates completely (sharpness 1.0), so the scales are both real and correctly indexed.

### 48.3 Layer order

The `layer` field in the weight map was inferred and never validated functionally. Identity 13.422,
reversed 13.883, block-reversed 13.217, even-odd 13.115 -- and **the best of five random orderings
scores 12.756, beating identity.** No ordering hypothesis is supported, but the control result is
itself informative: if the blocks were performing correct sequential computation, scrambling them
would be catastrophic. It is not. The blocks are largely interchangeable, which is what a stack of
mostly-noise-injecting layers looks like.

### 48.4 A correction: the OV metric is probably the wrong probe

Sections 43-45 treated "V/O at the OV noise floor" as *the* blocker. That reading does not survive
sec.47.2. Layers 3-13 produce **below-chance** NLL, and those layers contain V and O -- a completely
broken V/O could not yield a sub-chance truncated model. The consistent reading is that pico's OV
circuits do not perform embedding-copying at all, so the metric measures something this model simply
does not do, and its null was never evidence of a defect.

This does not restore V/O to "confirmed" -- it has no positive evidence either. It moves V/O from
"proven broken" back to "untested", and removes the justification for the large search effort spent
on it in sec.44-45 and sec.48.1.

### Standing

The +96 payload fix (sec.46) remains the session's one proven advance. Everything attempted since is
null: residual-basis solve, scale indexing, norm placement, embedding scale, layer order. The defect
remains cumulative and unlocated, and the most reliable handle on it is still the depth profile --
sub-chance through layer 13, degrading monotonically thereafter.

## 49. SIP re-enabled; the sub-chance result survives its control; layer composition is the fault

### 49.0 Environment change: static re-decoding is now blocked

Mid-session, `/System/Library/AssetsV2` became unreadable: `csrutil status` now reports **SIP
enabled** (it had been disabled). The pico asset's `binary_0.hwx` is **not** in the local
`FM_GenerativeModels_copy` mirror (that mirror holds the 3B adapters only), so no new *decode*
hypothesis -- payload offset, nibble order, codec, scale indexing -- can be tested until SIP is
disabled again. Everything in this section therefore runs on `pico_w96.npz`, the corrected +96
decode of all 7 roles x 24 layers, which remains fully sufficient for rearrangement and
architecture questions.

### 49.1 More architecture nulls

* **True post-norm.** Every prior norm test applied the extra RMSNorm to the sublayer *output*
  (`x = x + N(f(N(x)))`), leaving the residual stream unbounded -- it grows 1.19 -> 319. The
  untested alternative normalizes the *stream*: `x = N(x + f(N(x)))`. Measured: pre 13.372,
  resid-post-attn 13.623, resid-post-ffn 14.059, resid-post-both 14.531. **Pre-norm is correct**;
  the depth degradation is not a missing normalization.
* **Head layout in the full forward.** Q/K/V head split (head-major vs dim-major), O concatenation
  order, and GQA mapping (blocked vs interleaved) had only ever been checked against the OV metric.
  All 8 combinations in the forward: 13.098-14.016, none below chance.
* **Per-layer weight statistics.** Mean magnitude and stable rank are smooth across depth
  (early/late ratios 0.70-1.33, no discontinuity), so the layers are decoded consistently and there
  is no mis-mapping fingerprint.

### 49.2 The sub-chance result is real -- it passes its control

Section 47 reported layers 3-13 beating chance. That could have been an artifact: early in the
stack the residual still largely *is* the input embedding. Control = identical forward with every
weight replaced by a norm-matched Gaussian, embedding and tokenizer untouched:

| depth | decoded | norm-matched random |
|---|---|---|
| L3-L12 | **11.73 - 12.29 (below chance)** | 12.92 - 13.38 (never below) |
| L14-L23 | 12.93 - 13.48 | 12.82 - 13.19 |
| minimum | **11.728 @ L4** | 12.775 @ L0 |

**Random weights never beat chance at any depth.** The decoded weights do, from layer 3 through
layer 12. This is the first controlled, positive evidence that the decode carries genuine
language-modelling signal. Past layer 14 the decoded model becomes indistinguishable from random,
and slightly worse.

### 49.3 The fault is composition, not individual layers

Scoring each layer on its own -- fixed prefix (layers 0-3), then one candidate layer, against that
layer's own norm-matched random twin -- **12 of 20 layers beat their twin**, and layer 23 scores
**11.741** applied at depth 4, as good as layer 4 itself (11.807). Individual layers largely carry
signal; stacking them destroys it.

A greedy search over orderings (hold the residual, commit the layer minimising NLL at each depth)
was then **cross-validated on held-out text** of different content including non-Latin script,
because greedy over 24! orderings on one text is exactly the overfitting that killed the 3B
router-map candidates:

| order | train | **held-out** |
|---|---|---|
| identity (0..23) | 13.544 | 12.985 |
| greedy, full 24 | 11.738 | **12.437** |
| **greedy, depth 5** | 10.856 | **11.345** |
| identity, depth 5 | 11.929 | 12.084 |
| random orders, best of 6 | 13.118 | 13.029 |

**The greedy order transfers.** It beats identity and every random order on text it was not fitted
to, so the improvement is real rather than selection noise.

But the honest reading is *not* "the layer assignment is scrambled". The strongest configuration is
a **five-layer subset** (`10, 23, 0, 22, 17`) at 11.345 held-out, beating every 24-layer
arrangement. The reconstruction is at its best when nineteen of twenty-four layers are discarded.
That is a symptom, not a fix: it says the per-layer decode is still substantially wrong in a way
that compounds, and stacking more of it subtracts rather than adds.

### Standing

Established this session: the +96 payload fix (sec.46, proven structurally), and now a controlled
demonstration that the decode carries real signal (layers 3-12 beat chance where random weights
never do). Still open: pre-norm confirmed but the graph's third RMSNorm unplaced; V/O untested; the
per-layer defect that makes depth subtract. Best held-out NLL anywhere is 11.345 against a chance of
12.477 -- real, and still nowhere near the ~3-4 a working 300M model would give.

## 50. The asset was never actually lost, and +96 is confirmed on a second build

### 50.1 SIP is not a blocker: the models ship as mountable Cryptex images

Section 49.0 recorded SIP re-enablement as blocking all decode-level work. That was wrong in an
important way. `/Volumes/D/FM_GenerativeModels_copy` -- made long ago by `copy_apple_fm_models.sh`
-- contains the **Cryptex disk images**, not just metadata:

| image | size |
|---|---|
| `UC_FM_LANGUAGE_INSTRUCT_3B_BASE_GENERIC_SPARSE_GENERIC_H16G_IFP_Cryptex.dmg` | 5.4 GB |
| `UC_FM_LANGUAGE_INSTRUCT_3B_BASE_GENERIC_GENERIC_H16G_Cryptex.dmg` | 1.3 GB |
| **`UC_FM_LANGUAGE_INSTRUCT_300M_BASE_GENERIC_GENERIC_H16G_Cryptex.dmg`** | **426 MB** |
| `UC_FM_LANGUAGE_INSTRUCT_3B_IMAGE_ENCODER_*` (dense + sparse) | ~350 MB each |

These mount read-only with `hdiutil attach` and require no SIP change. The 300M image contains the
complete asset -- `binary_0.hwx`, `program.odix`, `program.dbginfo`, `manifest.plist`,
`specialized_model_0.mpsgraph`, and both LoRA constant blobs -- now copied permanently to
`local/pico_asset/` (406 MB).

**Caveat: it is a different build.** `binary_0.hwx` is 187,826,176 bytes here versus 193,921,024
in the SIP-protected copy, and `program.odix` 145,510,440 versus 136,052,680, while
`lora_32_constant_data.bin` is byte-identical at 29,687,808. `metadata.json` confirms the same model
(`afmplus-v11.0-pico`, `model_config: v11-pico`, `context_length: 4096`). So `pico_weight_map.json`
-- whose block offsets were read from the newer build -- does **not** transfer, and the map must be
re-derived (the symbol table parses cleanly: 29,383 symbols, 1,632 `_ne_0` tensor bases, `__KERN_0/1/2`
at the same VM addresses with different file offsets).

### 50.2 The +96 tile format is confirmed on an independent build

The differing build is an independent witness for sec.46, whose header claim was derived entirely
from the other file. Testing tiles resolvable through the second build's symbol table:

| check | result |
|---|---|
| sorted fp16 codebook @+0, zeros @+32, positive scales @+64, high-entropy @+96 | **700 / 775 (90.3%)** |
| final 32 bytes of the tile all zero | **680 / 775 (87.7%)** |

The ~10% that fail are `'s'` and `'L'` class tiles, for which this test wrongly assumed N-class
geometry (stride 0x2080, payload 8192). **The 96-byte header is confirmed across two independent
builds**, which also closes the gap noted earlier that the padding proof rested on N-class tiles of
a single file.

### 50.3 Full inventory of what Apple actually ships

**ANE models (Cryptex images, all present locally):** 300M base (pico), 3B base dense, 3B base
sparse/IFP, 3B image encoder (dense + sparse), 300M image tokenizer, a 9M event-extraction
classifier, and ~20 task-specific draft adapters (summarization, mail reply, machine translation,
proofreading, photos memories, shortcuts, smart reply, ...).

**Metal models (`model_type: mlm`, metadata only -- no weights on device):**

| display_version | layers | hidden | ffn | ctx | dtype |
|---|---|---|---|---|---|
| `afmplus-v7.0-150b` | 48 | 2048 | 5888 | 8192 | fp16 |
| `afmplus-v8.0-150b` | 48 | 2048 | 5888 | 8192 / 32768 | fp16 |
| **`afmplus-v9.0-3b-pcc`** | **56** | 2048 | **6656** | **32768** | fp16 |

These declare `backend: metal`, `data_type: fp16`, ASTC 6x6 compression in the `tamm_id`, and the
`afm_150k` sentencepiece tokenizer -- a completely different storage path from the ANE int4 tile
format, and one that would not have the ANE swizzle problem at all. Only their metadata is
downloaded; no `model.mlm` payload exists anywhere on the system, so they are not currently
attackable. The `-pcc` suffix indicates Private Cloud Compute, i.e. the server-side model, which is
consistent with it being the largest configuration (56 layers, 32k context) and with no weights
shipping to the device.

### Standing

Decode-level work is **unblocked without touching SIP**. The +96 header is now confirmed on two
independent builds. The immediate next step is to re-derive `pico_weight_map.json` against
`local/pico_asset/` so the corrected decode can be rebuilt on a build that is permanently available.

## 51. Apple's manifest confirms the architecture; block order is not the defect

With the asset permanently local (sec.50), `manifest.plist` -- previously unreadable -- was parsed.

### 51.1 The architecture, in Apple's own names

`Callables` holds 998 entries whose keys are full module paths with tensor shapes embedded. Collapsing
the layer index:

| module | shape | role |
|---|---|---|
| `attention_qkv_transform_..._lora_0` | 1x1024 -> 1x1024 | **Q** |
| `attention_qkv_transform_..._lora_1` | 1x1024 -> 1x256 | **K** |
| `attention_qkv_transform_..._lora_2` | 1x1024 -> 1x256 | **V** |
| `attention_output_transform` | 1x1024 -> 1x1024 | **O** |
| `feed_forward_hidden_transform_linear_0` | 1x1024 -> 1x3200 | **gate** |
| `feed_forward_hidden_transform_linear_1` | 1x1024 -> 1x3200 | **up** |
| `feed_forward_output_transform` | 1x3200 -> 1x1024 | **down** |

This is independent confirmation from the shipped asset of the seven-role decomposition and every
shape used throughout this work. It also settles two hypotheses on the record: **Q, K and V are three
separate transforms**, and **gate and up are two separate linears** -- so neither pair is fused or
interleaved in storage, consistent with the null results in sec.42.4 and the split sweep.

The names also carry per-shape specializations (`...x64f16` and `...x8f16`, i.e. sequence tiles of 64
and 8), which is why each module appears ~142 times rather than 24.

An attempt to bridge the manifest to the hwx by hashing failed: the Callable name suffixes decode as
16-byte base64url values, but only 28 of 920 appear anywhere in the hwx symbol table, and those are
2-4 hex-character coincidences. There is no direct name->symbol correspondence to exploit, so tensor
identity still has to be inferred from the symbol table order rather than read off.

### 51.2 The open assumption, stated by the decoder itself

`src/pico_weights.py` documents precisely what remains unproven:

> **ASSUMED** (a fixed ANE conv-layout convention, provably SV-invisible, NOT stated in the asset):
> the 16 tiles fill the block in row-major order; **blocks fill a multi-block `[Cout,Cin]` tensor in
> row-major order**; K/V are read `[1024,256]`. ... **Values are exact; only element->position may
> permute.**

That names the residual candidate exactly: block-to-output-range assignment.

### 51.3 Block order tested -- null

`O` and `down` are the two tensors whose *output* axis is the residual stream, each assembled from 4
blocks of 256 channels. If their block order disagreed with the embedding's channel order, every
layer would write into the wrong residual quarters -- exactly the cumulative signature. The space is
only 4! per tensor, so it was swept exhaustively and cross-validated:

| permutation | train | **held-out** |
|---|---|---|
| identity | 13.372 | 12.985 |
| best on train `(2,1,3,0)` | **12.348** | 12.979 |
| `(2,1,0,3)` | 12.586 | 12.813 |
| best `O` alone `(3,1,2,0)` | 12.709 | 13.122 |
| best `Q` blocks alone `(1,2,0,3)` | 12.710 | 12.880 |

The train-best gains 1.02 nats and then **returns 12.979 held-out against identity's 12.985 -- no
transfer at all**, and the train and held-out rankings disagree. With 24 candidates this is exactly
the overfitting profile seen before. **Block order is not the defect.**

### Standing

The seven-role decomposition and all tensor shapes are now confirmed from Apple's own manifest rather
than inferred. The remaining unknown is unchanged and now precisely quoted from the decoder's own
documentation -- an element->position permutation inside the ANE layout convention -- with tile
order, block order, head layout, orientation, and coarse permutations all eliminated against controls.

## 52. The decode is fully reproducible from the local build; two more families eliminated

### 52.1 The weight map re-derived, and the two builds proven identical

The per-layer block pattern documented in `pico_weights.py` was verified directly against the local
build's symbol table: 960 blocks resolve to file offsets, and their spacings classify themselves
(`N`=0x20800, `s`=0x10800, `L`=0x64800) into exactly `[N x10][s][N x12][s][N x12][L x4]` repeating --
**960 = 24 layers x 40 blocks**, with roles Q(4N) K(1N) V(1N) O(4N) gate(s+12N) up(s+12N) down(4L)
matching the manifest's module order.

Decoding layer 0 from this re-derived map and comparing against `pico_w96.npz` (decoded from the
other build):

| role | cos(flat) | max abs diff |
|---|---|---|
| Q, K, V, O, gate, up, down | **1.000000** | **0.00000** |

Bit-exact on all seven. **The two builds contain identical weights**, and the entire decode is now
reproducible from the permanently-local asset with no dependence on SIP or the protected copy.

### 52.2 Input-chunk width: C=1 confirmed

ANE convolution weights are stored in chunks along the input axis. The slot map

```
chunk = slot // (nout*C) ; rem = slot % (nout*C) ; o = rem // C ; i = chunk*C + rem % C
```

reduces to the current decode at `C=1` and to the already-refuted `imin` at `C=ncin`. Every
intermediate value is a distinct plausible ANE layout, and the earlier sweep had only ever probed
the two endpoints. Layer-0 attention sharpness across the family:

| C | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sharp | **0.6058** | 0.083 | 0.023 | 0.015 | 0.013 | 0.009 | 0.014 | 0.011 | 0.010 | 0.014 | 0.011 |

`C=1` wins by 7x over the nearest competitor. **The intra-bank slot map is confirmed.**

### 52.3 Embedding residual basis: a promising candidate that did not survive

The embedding was recovered by **dynamic memory capture** with its own de-swizzle; the weights come
from the hwx. Two independent extraction paths, never cross-checked for channel order -- and the
semantic validation (Paris -> PARIS/Paris(fr)/Parizh) is blind to a permutation of the 1024
*columns*. Only the relative permutation matters, so applying pi to E alone is the test.

`revblk(64)` (reverse the order of 16 blocks of 64 = head_dim) initially looked strong: train 12.231
/ held-out 12.386 against identity's 13.478 / 12.985, beating all 20 random controls held-out, and it
appeared to **remove the depth degradation** entirely (identity decays 11.81@L4 -> 13.42@L23;
revblk(64) holds 11.65-12.22 through L23).

**Both signals failed their controls.**

* The held-out advantage is seed-dependent. With one control sample the random minimum was 12.614
  (revblk(64) beat all 20); with another it was 12.248, putting revblk(64) *inside* the random tail.
* The depth-flattening is a **confound, not a property**. Random permutations produce equally flat or
  flatter profiles -- drift `L23-L4` of -0.41, -0.22 and **-1.05** versus revblk(64)'s -0.08 -- because
  a permutation with poor early NLL simply regresses toward the ~13 plateau and looks flat, while
  identity's good early score has room to decay. **Depth drift is confounded by starting NLL and is
  not a valid instrument.** It should not be used again without pairing it to absolute NLL.

`revblk(64)` is therefore withdrawn, along with the claim that the depth degradation was fixed.

### Standing

Reproducibility is now complete and independent of SIP: block pattern verified, map re-derived,
decode bit-exact against the other build. Confirmed this section: `C=1` intra-bank slot map.
Eliminated: the input-chunk family, and the embedding-basis permutation family (no member survives
controls). The defect remains an element->position permutation that no tested family contains.

## 53. The FFN relative permutation, re-swept on corrected weights: a candidate that survives

### 53.1 Why this needed redoing

Section 42.3 proved the FFN has exactly one real degree of freedom -- the permutation of down's
input axis *relative* to gate/up's output channels (any joint permutation is a no-op). It was swept
in sec.42.4 and again in sec.43.3, but **both ran on the +128 decode that sec.46 proved wrong**, so
both were void. This re-runs it on `pico_w96.npz` with held-out validation the earlier sweeps lacked.

### 53.2 Identity is not the answer

69 structured bijections of 3200, scored on train with 16 controls:

| pi (down input axis) | train | ctl beating | held-out | ctl beating |
|---|---|---|---|---|
| `revblk(5)` | **12.255** | 0/16 | **12.526** | 0/16 |
| `T(160,20)+rev` | 12.340 | 1/16 | 12.404 | 0/16 |
| `revblk(10)` | 12.626 | 3/16 | 12.593 | 0/16 |
| **`identity`** | 13.515 | 12/16 | 12.985 | 6/16 |

**Identity ranks 52 of 69.** The current down input order is worse than most alternatives -- itself
evidence that this axis is wrong.

### 53.3 `revblk(5)` survives confirmation and its confound control

Confirmatory test on a **third** text set never used in selection, with 40 controls each:

| pi | set1 | ctl< | set2 | ctl< | set3 | ctl< |
|---|---|---|---|---|---|---|
| `identity` | 13.515 | 35 | 12.985 | 17 | 12.935 | 15 |
| **`revblk(5)`** | **12.255** | **0** | **12.526** | **0** | **12.334** | **2** |

Mapping the whole reversal family across all three sets gives a sharp spike rather than a smooth
trend: rank-sum 2 for `g=5`, 13 for `g=10`, and >=22 for every other group size including the pure
full reversal `g=1`, which is *bad*. So the 5-grouping is essential, not incidental.

**The obvious confound is excluded.** A permutation that merely attenuates a harmful FFN would
improve NLL on every text without being correct. Measured:

* `||f||/||x||` is **unchanged**: 0.3538 (identity) vs 0.3606 (`revblk(5)`) -- no muting.
* A pure scalar attenuation of down never helps at all: alpha=1.0 is optimal (12.985 / 12.935) and
  every smaller alpha is monotonically worse, down to 14.14 / 14.16 at alpha=0.

So the gain is not attenuation, and it is not selection noise on one text.

### 53.4 Honest assessment

This is the strongest surviving candidate the project has produced -- the only one to clear
control bands on three independent text sets *and* its confound control. Two cautions stand:

1. **It is structurally implausible.** 3200 = 5 x 640, but nothing in the tile geometry (128, 256,
   16 banks, 13 blocks) has a factor-5 period, and no ANE layout would naturally reverse 640 groups
   of 5. A real hardware permutation should have a power-of-two or block-geometry period.
2. **It does not make the model work.** 12.33-12.53 against a chance of 12.477 is a fraction of a
   nat, where a functioning 300M model would be ~3-4. Generation is still incoherent.

The defensible conclusion is narrower than "the FFN permutation is `revblk(5)`": **down's input axis
is demonstrably not in gate/up's channel order**, identity is measurably poor, and `revblk(5)` is
whatever structured permutation happens to sit closest to the truth among those tested. It is a
signpost toward the right axis, not the answer.

## 54. RETRACTION: revblk(5) is not a layout fact, and the NLL oracle has no power on this axis

### 54.1 The test that killed it

Section 53 presented `revblk(5)` as the strongest candidate the project had produced: it beat 40
controls on three independent text sets and survived the attenuation confound. That was not enough,
and the decisive test is one sec.53 did not run.

**A real storage-order correction must hold for every layer** -- the ANE lays all 24 out identically.
Applying `revblk(5)` to down's input axis in **one layer at a time**:

| | layers improved (of 24) |
|---|---|
| `revblk(5)`, set2 | **14** |
| `revblk(5)`, set3 | **14** |
| a **random** permutation, set2 | **14** |

Per layer, `revblk(5)` is **indistinguishable from a random permutation**. Individual deltas scatter
from -0.38 to +0.39 with no consistency between the two text sets (layer 1: +0.06 vs +0.30; layer 3:
-0.31 vs -0.12; layer 23: +0.25 vs +0.34). The all-layers gain (12.985 -> 12.526) is 24 scattered
accidents composing favourably, not a systematic correction.

**`revblk(5)` is withdrawn.** So is the framing of sec.53.4 that called it "a signpost toward the
right axis" -- it points nowhere.

### 54.2 The larger consequence: this oracle cannot decide this axis

The random-permutation row is the important one. Perturbing a single layer's down input axis at
random **improves the model half the time**. That is what it looks like when a tensor's contribution
is already close to noise: there is no gradient of correctness to detect, so every permutation scores
within the same band and rankings among them are meaningless.

This retroactively explains the whole sequence of FFN sweeps -- sec.42.4, sec.43.3, sec.53 -- in
which candidates kept clearing control bands and then failing to mean anything. **The NLL oracle has
no power on the FFN axis**, and the correct response is to stop running permutation sweeps against it
rather than to run better-controlled ones.

The `identity is bad` observation from sec.53.2 also loses its force: if all permutations score
within one noise band, identity's rank of 52/69 is not evidence about identity.

### 54.3 What still stands

Unaffected, because none of it rests on permutation search against NLL:

* the +96 tile format, proven structurally and confirmed on two independent builds (sec.46, sec.50)
* the C=1 slot map, which beats every chunk width by 7x on attention sharpness (sec.52)
* the weight map, re-derived and **bit-exact** against the other build (sec.52)
* the seven-role architecture and all shapes, confirmed from Apple's manifest (sec.51)
* the embedding and tokenizer (sec.43.1)
* layers 3-12 beating chance where norm-matched random weights never do (sec.49.2)

### Standing

The codec is solved and cross-validated. The ordering is not, and this section establishes that it
**cannot be solved by the instrument in use**: statistical scoring of permutations against a forward
whose FFN is already near-noise. Every candidate this method has produced -- the composition oracle's
winners, `T(4,800)+revblk`, `revblk(64)`, `revblk(5)` -- has died on contact with a sharper control.

That is the case for ground-truth activations. With real intermediate states captured from the live
model, the permutation stops being a search scored by a powerless oracle and becomes a direct
match between a measured tensor and a computed one.

## 55. The privileged capture, attempted: two infrastructure defects found

SIP was disabled and the capture run. It produced no activations, and diagnosing why invalidated
a piece of tooling this project had been treating as available.

### 55.1 `capture_pico_logits.sh` has never worked

Two independent defects, both fatal:

**(a) lldb deadlock.** The script sets `debugger.SetAsync(False)`, arms the breakpoint with
`SetAutoContinue(True)`, then calls `proc.Continue()` *before* starting the thread that drives the
oracle. In synchronous mode `Continue()` returns only when the process stops -- and auto-continue
means it never stops. So `Continue()` blocks forever and the oracle is never launched. Observed
directly: 27 minutes at 0% CPU with an empty `binds.jsonl`. Fixed by setting `SetAsync(True)` and
starting the driver thread first.

**(b) The wrong hook entirely.** The script breakpoints `_espresso_network_bind_buffer`. **pico is
an `.mpsgraphpackage` driven through MetalPerformanceShadersGraph, not Espresso**, so that symbol
is never called on this path and the breakpoint cannot fire regardless of which process is
attached. After fixing (a), the run completed cleanly and `binds.jsonl` was still empty.

The corroborating evidence was already in hand and had not been read: the existing
`local/pico_oracle/` directory contains `binds.jsonl`, `greedy_token.txt` and a core -- **but zero
`.f16` files**. The buffer capture never produced data on any previous run either. The "functional
oracle" described in that script's header, and cited in earlier sections as an available
instrument, has never actually run.

### 55.2 The core fallback is mistimed

The fallback launches the driver, sleeps 2 seconds, then dumps. `afm` spends those seconds loading
the model, so the 9.72 GB core (118 segments, from the process with the real RSS) captures model
*loading*, not inference. Measured on that core:

* exact fp16 byte match for five known embedding rows: **0 hits**
* best |cosine| against those rows over the entire file, at two alignments, scale-invariant so it
  would catch a scaled or normalised copy: **0.18 - 0.22**, i.e. noise

So the core contains no activations at all -- neither the input embeddings nor anything derived
from them.

### 55.3 What the run did establish

* The model itself is healthy and its answer is known: `afm --temp 0 --max 12 "The capital of
  France is"` returns `{"capital": "Paris"}`. Output is JSON-constrained because the asset ships a
  `constraints_override` schema. The real flags are `--system/--temp/--max/--stream/--server`; the
  `-t 0 -m 1` used by the old script was never parsed.
* `E` is not implicated by the null result. The pico embedding is decoded from `program.odix` and
  was already validated **bit-exactly** against independently captured rows; its absence from this
  core is a property of the core, not of the decode.

### 55.4 The fix

`capture_pico_core2.sh` drops the breakpoint entirely -- no symbol assumptions -- and instead
drives a **long** generation (`--max 400`), polls until a service process is genuinely hot
(>=15% CPU), and only then dumps, twice, a few seconds apart. Two snapshots inside one generation
let an offline scan separate per-token activations from static buffers.

### Standing

No ground-truth data yet. The obstacle turned out not to be SIP but two latent bugs in the capture
tooling, one of which means a documented instrument never existed. That is worth knowing: several
earlier sections treated "a functional oracle is available if needed" as a fallback, and it was not.

## 56. Three cores, 28 GB, wrong process: TG never sees the model

`capture_pico_core2.sh` fixed the timing problem -- it drove a long generation and dumped only once
a service exceeded 15% CPU, twice, seconds apart. Both cores came from
`TGOnDeviceInferenceProviderService` (pid 1149), the process the CPU trigger selected. Neither
contains activations, and the reason is now established: **that process never sees the model at all.**

### 56.1 Byte-level presence tests

Exact byte search for high-entropy 2 KB/4 KB needles drawn from each shipped component, across both
mid-inference cores (8.9 GB + 8.5 GB):

| component | form | present in TG core? |
|---|---|---|
| `binary_0.hwx` (ANE program, 187 MB) | ANE-swizzled int4 | **absent** (3/3 needles) |
| `program.odix` (CPU program, 145 MB) | int4 + fp16 tables | **absent** (3/3 needles) |
| `lora_32_constant_data.bin` (29.7 MB) | **plain fp16** | **absent** (3/3 needles) |

The LoRA row is the decisive one: it is plain fp16, not swizzled and not quantised, so it is the
component most likely to be handled CPU-side. Its absence means **no model data of any kind** is
mapped into this process.

Activation scans agree. Cosine against the 42 token embeddings actually used in this run,
scale-invariant and at two alignments, over both cores: best `|cos| = 0.365` with the per-chunk
median already at 0.300 -- i.e. indistinguishable from noise.

### 56.2 What TG actually is

Process facts: `TGOnDeviceInferenceProviderService` runs as `_modelmanagerd` with 68 MB RSS;
`IntelligencePlatformComputeService` runs as `euler` with 10 MB; `aned` runs as root with 3 MB. None
is large enough to hold a 187 MB program, and the byte tests confirm TG does not map it. TG is an
XPC broker: prompts in, tokens out. The earlier assumption -- inherited from the 3B work, where TG
cores did yield fp16 rows -- does not transfer to pico.

### 56.3 Status of the capture programme

Three privileged captures have now been run. They produced:

* the two tooling defects of sec.55 (deadlock; Espresso hook on an MPSGraph model), both real bugs
  in code this project had documented as a working oracle
* a fixed, timing-correct capture script
* proof that the capture *target* was also wrong

None produced ground-truth activations. The next step is not another blind dump but a five-second
`lsof` on the asset path to identify which process actually opens it -- and if the answer is none,
then the model is mapped straight into ANE device memory by the kernel driver and **no user-process
core can ever contain it**, which would close the dynamic-capture route for pico on structural
grounds rather than for want of privilege.

### Standing

SIP is not the obstacle and never was. The obstacle has been a chain of incorrect assumptions in the
capture tooling -- wrong synchronisation, wrong hook, wrong process -- each of which had to be
falsified by measurement. That work is not wasted: the capture path is now correct except for the
target, and the target is one command away.

## 57. The capture target found, and the dynamic route closed on evidence

`lsof` on the asset paths resolved the confusion of sec.56 immediately:

| pid | asset held |
|---|---|
| 1149 | **3B** -- `56659e51.../main-h16g.odix` + `ifp_rasterized_weights.bin` |
| **2445** | **pico** -- `031c7be6.../program.odix` |

All three earlier captures dumped 1149. pico lives in a *second* instance of the same service
binary, which sits at 3.4 MB RSS and never goes hot because it is the **speculative-decoding draft
model** -- so the CPU trigger could never select it. Selecting by open asset instead, and dumping
2445 during a generation, its RSS rose 3.4 MB -> 71 MB, i.e. it demand-paged the model in.

### 57.1 Right process, confirmed

| component | present in pid 2445's core? |
|---|---|
| `program.odix` (CPU program, 145 MB) | **FOUND @0xfb4af60** (both cores) |
| `binary_0.hwx` (ANE program) | absent |
| `lora_32_constant_data.bin` (plain fp16) | absent |

So the target is finally correct: pico's CPU program is mapped here. The ANE program's absence is
expected -- the kernel driver stages it into ANE device memory.

### 57.2 But the activations are not in user space

Scale-invariant cosine against the 46 token embeddings actually used in this run, two alignments,
both 7.5 GB cores: **best |cos| = 0.365**, identical to the wrong-process cores and to pure noise.
`program.odix` is mapped **as a read-only file** -- the asset itself, which was already on disk --
not as live state.

This is not a sampling accident. Each captured generation emitted ~40 tokens, and speculative
decoding invokes the draft model for *every* token, so pico executed well over a hundred forward
passes inside the capture window. If its activations reached CPU memory they would be present.

### 57.3 Verdict on the capture programme

Four privileged captures, five cores, ~40 GB. What they produced:

* **two real bugs** in `capture_pico_logits.sh` -- an lldb deadlock, and an Espresso hook on a model
  driven through MPSGraph -- in code this project had documented and cited as a working oracle. It
  had never run. (sec.55)
* proof that the capture *target* was wrong, then identification of the right one (sec.56, above)
* **the finding that matters**: pico's intermediate activations never enter user-process memory.
  They live in ANE device memory, which `save-core` cannot reach.

**The dynamic-capture route is closed for pico on structural grounds, not for want of privilege.**
SIP can be re-enabled; it was never the obstacle, and disabling it did not and could not help.

### Standing, corrected

Section 54 argued that ground-truth activations were the way past the powerless NLL oracle. That
remedy is now measured to be unavailable by this method. The honest position: the ordering cannot be
determined by permutation search (sec.54), and cannot be measured by core capture (this section).
What remains untried is intercepting at the ANE boundary itself -- the driver's IOSurface handoff --
which is a materially harder instrument than anything attempted here, and should not be presented as
a near-term fix.
