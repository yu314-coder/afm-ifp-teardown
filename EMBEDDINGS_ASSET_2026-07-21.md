# The 3B embedding is now SHIPPED — a new asset overturns Finding "3B embedding is ANE-locked"

Discovered 2026-07-21 after a macOS update (27.0, build **26A5378n**). The
`com.apple.MobileAsset.UAF.FM.GenerativeModels` catalog grew from **118 to 124 assets**; six new
assets were downloaded at 03:00 that day. One of them changes the project's headline negative result.

## 1. The asset

```
UC_FM_LANGUAGE_INSTRUCT_3B_EMBEDDINGS_GENERIC_GENERIC_H16G_IFP_Cryptex.dmg    (285,212,672 B)
```

It is a mountable read-only cryptex (`hdiutil attach -readonly -nobrowse -noverify`, no root needed).
Contents:

```
metadata.json                          {"vocab_size": 262144, "embedding_dim": 2048,
                                        "embeddings_signature": "db933c00..."}
model.odixpackage/main-h16g.odix       268,442,456 B
model.odixpackage/main-h16g.dbginfo
```

**The size arithmetic is exact.** `262144 x 2048` elements at 4 bits = **268,435,456 B**, and the file
is 268,442,456 B — a difference of exactly **7000 bytes** of header. The header is an `odix`
flatbuffer whose strings include **`load_embeddings`**, `outputs`, `embeddings`,
`exec.coreml_model`. Payload begins at offset 7000.

## 2. Why this matters

The paper's Finding `find:embed` / `find:3bane` states that the 3B token embedding is **not present in
the shipped assets at all** — ~1.14M candidate offsets across the weight file scored <= +0.047 against
a probe calibrated at +0.53, and an 8.2 GB full-memory core contained zero D=1536 buffers. That was
correct **for the assets shipped at the time**. It is no longer true of the current catalog: Apple now
ships the embedding as a **separate, static, purpose-built asset**.

Note the dimension: **2048**, not the sparse IFP backbone's 1536. This matches
`afmplus-v11.0-nano` (the ~3B dense backbone, 2048-wide), so the asset serves the dense 3B.

The gap therefore changes category — from an **information limit** ("the data is not in the package")
to a **decode problem** ("the data is in the package in a layout not yet identified").

## 3. What the payload is

Nibble histogram over the first 4 MB, by code value:

```
code:   0     1     2     3     4    5    6    7    8    9    10   11   12   13    14    15
pct: 29.74 17.45 12.17 3.27 1.48 0.37 0.23 0.25 0.00 0.24 0.21 0.42 1.53 3.22 12.15 17.27
```

This is **symmetric about zero under a signed int4 reading** (0 most common; +-1 at 17.4/17.3;
+-2 at 12.2/12.2; code 8 = -8 entirely unused). So the payload is a dense **linear signed int4**
tensor, not a palettized one — no codebook is required. A scan of the 7000-byte header found **no
fp16 scale array** (no run of >=128 plausible fp16 values), so no per-dim scale ships with it.

## 4. The layout is NOT yet identified — with a validated oracle

Scoring uses the calibrated orthographic-pair oracle (singular/plural cosine minus an id-matched
control). **The oracle was validated first**, on pico's known-good, independently-verified embedding
using the identical code path:

```
pico control:  cos(dog,dogs)=+0.648  cos(king,kings)=+0.664  ...
               mean orthographic +0.6219 | mean control +0.1654 | DELTA +0.4565
```

So the instrument has power, and the following negatives are real. Roughly 20 layout hypotheses were
tested; **all sit at the noise floor** (|DELTA| <= 0.054 against a genuine ~+0.46):

| family | best DELTA |
|---|---|
| row-major `[V,D]`, signed and unsigned | +0.013 |
| pico-style n-way token interleave with lane skew (n = 1,2,4,8,16; several skews) | +0.054 |
| column-major / transposed `[D,V]` | +0.007 |
| dim-pair and token-pair swaps | +0.003 |
| the proven 3B ANE 8x128 de-swizzle (IF = 8,16; CH = 64,128,256; both row dims) | +0.005 |

Structural probes are equally flat: no zero runs >= 64 B anywhere in a 16 MB sample (so no unused
vocab slots to key on), no stride stands out in a block-variance profile at 1024/2048/2064/4096 B,
and byte autocorrelation peaks at **2064**, which corresponds to no obvious `[V,D]` row stride.

## 5. Status

The 3B embedding **is now on disk, in a file of exactly the right size, in a plausible signed-int4
encoding, in an `odix` program whose header names `load_embeddings`** — but it has not yet been read
out. The remaining work is to recover its element layout, which is a decode problem with a working
oracle rather than the information wall previously reported.

Next step: parse the 7000-byte `odix` flatbuffer header properly (the project already has an odix IR
grammar) to obtain the tensor descriptor directly, rather than continuing to guess layouts.

No Apple weights are committed; this records only offsets, statistics, and findings.

## 6. Further layout attempts, and a methodological correction

**The odix header carries the tensor descriptor.** Parsing the 7000-byte flatbuffer locates an
`NDArray` descriptor near offset `0x1a30` holding, as consecutive int64s,
`262144, 1, 2048` — i.e. shape **`[V, 1, D]`**, repeated again at `0x1ac8`. So `[V, D]` is confirmed
as the *logical* shape. Nearby strings: `IOSurface`, `Context.alloc`, `$load_embeddings`.

**A correction that invalidates part of the sweep above.** Cosine similarity is invariant under any
*fixed permutation applied to both operands*. The orthographic oracle therefore **cannot see
dimension order at all** — it only tests whether the right *set* of nibbles is grouped into a token's
row. This was visible in the data: low-nibble-first and high-nibble-first produced byte-identical
scores. So the many "layouts" tried above collapse to far fewer distinct hypotheses, and the failure
means the **row grouping** is wrong, not the intra-row order.

Additional groupings tested, all at noise: ANE plane format (`dim` blocked by CB = 8, 16, 32, 64, 128,
both channel-fast and width-fast), and nibble-order x zero-point variants (signed two's-complement,
zero-point-8, raw). Best DELTA across all of these: **+0.020** against a genuine +0.46.

**A tokenizer-free discriminator** (distribution of cosines over 400 random token pairs, calibrated on
pico) is more informative than the paired oracle here:

| source | mean | std | p95 | max |
|---|---|---|---|---|
| **pico (known-good control)** | +0.340 | 0.121 | +0.550 | **+0.988** |
| 8-lane interleave | +0.114 | 0.101 | +0.237 | **+0.936** |
| DV transposed | −0.004 | 0.168 | +0.278 | +0.538 |
| plane CB=8 | +0.002 | 0.155 | +0.269 | +0.471 |
| VD row-major | +0.011 | 0.043 | +0.104 | +0.341 |

The 8-lane interleave — pico's own layout family — is the only candidate reproducing the
**near-duplicate rows** (max cosine 0.94, cf. pico 0.99) characteristic of a real embedding table;
plain row-major has no near-duplicates at all, which is itself evidence against it.

**But the id mapping does not match pico.** A nearest-neighbour agreement test over a 600-token
sample (does each token have the same nearest neighbour in both files?) gives 0.50% for the 8-lane
layout, 1.17% row-major, 0.33% transposed, against 0.17% chance — no meaningful agreement, and
top-10 neighbour overlap of 0.8–3.9%. So either the row grouping is still wrong, or this asset uses a
**different vocabulary ordering** than the `tok_vocab.json` derived from the IFP/pico models. The
latter is plausible: `embedding_dim = 2048` identifies this as the *nano* dense backbone, a different
model from the 1536-wide IFP sparse one.

**Bottom line unchanged from §5, with the search space now better characterised:** the file is
present, correctly sized, and plausibly encoded, and one layout family (8-lane interleave) shows the
right statistical signature — but the embedding has not been read out, and it cannot be validated
against a semantic oracle until the vocabulary ordering for this asset is established.

## 7. A real but partial layout signal, via a cross-model Gram test

The paired oracle is the wrong instrument for *searching* this space (§6: it cannot see dimension
order). A **cross-model Gram-matrix test** is far better, and it is dimension-independent, so pico's
$1024$-wide embedding can be used as a reference for a $2048$-wide one: take $N$ token ids, build each
model's $N\times N$ cosine matrix, and correlate the off-diagonals. Same tokenizer plus correct row
grouping should give a positive correlation; the null is a shuffled id mapping.

**Result for the 8-lane interleave (pico's own layout family):**

| test | value |
|---|---|
| raw Gram correlation vs pico | **+0.245** |
| shuffled-id null (20 draws) | −0.003 ± 0.029 (max +0.046) |
| after removing row-norm effects | +0.251 |
| after removing row-norms **and** token-id distance | **+0.245** |
| Spearman (rank-based, outlier-robust) | **+0.300** |
| Pearson excluding the top 10% of \|values\| | +0.244 |

That is roughly **8σ above the null** and survives every confound tested: row norms, token-id
locality (pico's own id-locality is only +0.055, far too small to explain it), rank transformation,
and outlier removal. By contrast plain row-major scores **+0.054** on the same test.

So the tokenizer **is** shared with pico, the token ids **are** aligned, and the 8-lane interleave
recovers genuine embedding structure. Other families tested and rejected on the same instrument:
chunked-vocab interleave and chunked token-major (all ≤ +0.042, at null), and `lane = t mod L`
dim-major for L = 2, 4, 16, 32, 64 (all below L = 8).

**But it is not the full answer.** The same layout scores **−0.011** on the orthographic oracle, which
does have power (+0.46 on pico). A layout that is entirely correct must score on both. The pattern —
coarse cross-model geometry preserved, fine morphological structure absent — is what one expects if
each recovered row is a **mixture** that includes the right token's elements alongside wrong ones:
dilution destroys the singular/plural signal long before it destroys bulk geometry.

**Where that leaves it.** The row grouping is近 but not exact; the residual error is finer than the
lane structure. The Gram test is the right objective to continue optimizing against, since it has a
measured null, ~8σ of headroom, and is robust to the confounds that made earlier instruments useless.

## 8. The other five new assets

For completeness, the rest of the 2026-07-21 delta (none carries an embedding table):

| asset | size | contents |
|---|---|---|
| `..._3B_OPEN_ENDED_EXTRACT_DRAFT_..._SPARSE_...` | 52 MB | speculative-decoding draft, `draft_steps 5`, ctx 8192; 36 MB `program.odix` + 10 MB `binary_0.hwx` |
| `..._300M_OPEN_ENDED_EXTRACT_DRAFT_...` | 58 MB | **despite the name, `model_config: v11-9m`** — a 9M draft, ctx 4096, `draft_steps 7`; 34 MB odix + 19 MB hwx |
| `..._3B_LW_PLANNER_V1_DRAFT_..._SPARSE_...` | 60 MB | planner draft, `draft_steps 13`, ctx 8192; 35 MB odix + 20 MB hwx |
| adapter `9b2eb1f8` | 110 MB | `lora_48` for `afmplus-v11.1-ifp`, `backbone_signature 0a6fe237…` |
| adapter `e9e1cee8` | 28 MB | `lora_32`, `backbone_signature cc4da08e…` |

Two notes. The `300M`-named asset is a **9M** model by its own `model_config`, so the filename refers
to the target it drafts for, not its own size — worth remembering when sizing assets by name. And all
four cryptexes are `H16G`, consistent with the architecture string that `ANECCompile` accepts.

## 9. The row geometry is solved; an 8-lane structure is proven

Three independent results pin most of the layout.

### 9.1 Row size and id order, confirmed by a cross-model coincidence

Hashing the payload at 32-byte granularity finds one block repeated **4608** times, and those
occurrences are **contiguous at the very end of the file**: the final $4608\times32 = 147{,}456$ bytes
are a single repeated pattern. That is exactly **144 rows of 1024 bytes**.

Independently, pico's captured logit vector has exactly **144 masked entries** (live vocab 262000).
The two numbers agree, which simultaneously establishes that

- the row stride is **1024 bytes** = 2048 int4 values, i.e. one row per token;
- token ids map to file position with the **identity** ordering (padding sits at the tail, ids
  262000–262143);
- the trailing 144 vocab slots are unused in both models.

### 9.2 Duplicate-group analysis proves an 8-lane structure

Hashing all 262144 contiguous rows yields 332 duplicate groups (unused/untrained vocab slots).
Every large group is **100% pure in `id mod 8`** and only ~50% pure mod 16:

```
group   size   purity of (id mod M):  M=2     M=4     M=8     M=16    M=32
  #0     748                          100%    100%    100%     50%     25%
  #1     747                          100%    100%    100%     50%     25%
  #2     733                          100%    100%    100%     50%     25%
  #3     707                          100%    100%    100%     50%     25%
```

So tokens sharing a value only decode identically when they share `t mod 8`. The lane count is
exactly **8**.

### 9.3 The lane structure, confirmed functionally against pico

Splitting the cross-model Gram test by lane gives an unambiguous result:

| pairs | n | Gram corr vs pico |
|---|---|---|
| **same-lane** | 50428 | **+0.3069** |
| **cross-lane** | 354122 | **−0.0188** |

and every individual lane is positive on its own — +0.313, +0.432, +0.315, +0.175, +0.314, +0.422,
+0.241, +0.373 for lanes 0–7. That is eight independent confirmations. It also explains the whole
earlier puzzle: only 1/8 of random pairs share a lane, so the pooled signal was diluted to +0.245,
and the orthographic oracle's test pairs (which mostly straddle lanes) read exactly zero.

**Reading a contiguous 1024-byte row therefore recovers a token's values in a form that is internally
consistent within a lane but not comparable across lanes.**

### 9.4 A correction

An earlier check in this session claimed the four largest duplicate groups were byte-multiset
identical, i.e. permutations of one vector. That used a Python `set()` comparison, which tests only
which distinct byte values occur — nearly vacuous over 1024 bytes. Re-checked with true
`bincount` multisets, they are **not** identical (e.g. code-0 counts 136/216/184/152), so those four
groups are four *different* unused vectors that happen to be lane-pure. The lane result in §9.2–9.3
is unaffected: it rests on group purity and the Gram split, not on that claim.

### 9.5 Remaining gap

Extracting with an explicit 8-lane interleave (`pos = (t/8)(8D) + 8j + (t mod 8)`) **removes the lane
asymmetry** — same-lane +0.150, cross-lane +0.142, i.e. finally non-zero across lanes — but at a lower
magnitude than contiguous same-lane (+0.307). A dim-blocked variant recovers +0.319 same-lane but
leaves cross-lane at +0.018.

So the true mapping is close to an 8-way interleave but not exactly the naive one: the correct form
must reach ~+0.31 **uniformly** across both same- and cross-lane pairs. That is now a
one-dimensional question — the relative dim-order between lanes — with a sharp, cheap objective
(cross-lane Gram correlation, currently 0 for the right-magnitude layouts and +0.14 for the
lane-symmetric one).

## 10. RETRACTION: the Gram evidence was built on an inadequate null

**§7's "~8σ" result and the token-alignment conclusion drawn from it are withdrawn.**

The cross-model Gram test used a **shuffled-id** null. Shuffling destroys the vocabulary's *regional*
structure completely, so any correlation arising from region membership — rather than token identity —
appeared as signal. The correct null is a **shift**, which preserves regional structure while
destroying token correspondence.

Under a shift null the signal vanishes:

```
shift      0      8      16     64     256    1024   4096   16384  40000
corr    +0.245 +0.308 +0.225 +0.204 +0.215 +0.245 +0.197 +0.093 +0.003
```

There is **no peak at shift 0** — shift 8 scores *higher* than the true ids — and the correlation
decays smoothly with shift magnitude, reaching zero only at ~40000. That is exactly what regional
structure produces: nearby ids occupy similar vocabulary regions (byte-fallback, control tokens,
frequent words, rare words), and both models' cosine matrices reflect region membership.

A directly equivalent check: substituting pico ids shifted by +1000 into the same-lane test scores
**+0.3435** against **+0.3708** for the true ids — indistinguishable.

**Consequences.** The following do NOT stand: that the tokenizer/id alignment was confirmed by Gram;
that the 8-lane interleave "recovers genuine embedding structure"; that same-lane extraction yields
correct token vectors. The +0.307 same-lane vs −0.019 cross-lane *asymmetry* is still a real
observation, but it shows only that cross-lane rows are mutually incomparable — not that same-lane
rows are correct.

This is the fourth instrument in this project to produce a confident-looking result with an
inadequate control (after "self-ranks 0", the single-token rank oracle, and the weight-statistics
alignment test). The recurring failure mode is identical each time: **a null that destroys more
structure than the hypothesis under test**. A null must differ from the hypothesis in exactly one
respect — here, token identity — and preserve everything else.

## 11. What actually stands

Purely structural results, derived from exact hashing and byte arithmetic rather than statistics:

1. **The asset exists and is correctly sized.** `262144 × 2048` int4 = 268,435,456 B, plus a 7000-byte
   `odix` header naming `$load_embeddings`, with an `NDArray` descriptor of `[262144, 1, 2048]`.
2. **Row stride is 1024 bytes and the id order is identity.** The file's final 147,456 bytes are one
   repeated pattern = exactly 144 rows, and pico's captured logits have exactly 144 masked entries
   (live vocab 262000). Two independent sources agreeing on 144 fixes both facts.
3. **The payload is linear signed int4** — histogram symmetric about zero, code 8 (−8) unused, no
   codebook, no fp16 scale array in the header.
4. **There is an 8-lane structure.** Hashing all 262144 contiguous rows gives 332 duplicate groups;
   every large group is **100% pure in `id mod 8`** and ~50% pure mod 16. This is exact-match
   evidence, independent of any statistical oracle.

**Not established:** that any extraction tried so far yields correct per-token vectors. The
orthographic oracle — the one instrument here with a validated positive control (+0.46 on pico) —
reads zero for every layout tested, and that remains the honest bottom line.

## 12. A trustworthy model-internal oracle, and what it shows

The retraction in §10 left no usable instrument: the paired oracle is blind to dimension order, and
the cross-model Gram test measures vocabulary regions rather than token identity. This section builds
one that has neither defect.

**The instrument.** Harvest morphological pairs directly from the vocabulary — every token `▁X` whose
inflection `▁Xs`, `▁Xes`, `▁Xing`, `▁Xed`, `▁Xly`, `▁Xer` is also a token — giving **17,674 pairs**.
Score = mean cosine over pairs minus mean cosine over an id-matched control (each `a` paired with a
different pair's `b`). This is **entirely internal to the 3B file**: no cross-model assumption, so the
shift-null failure mode cannot arise. Validated on pico over the same pair set: **DELTA +0.3487**.

**Result.** Splitting by lane:

| rows | same-lane pairs | cross-lane pairs |
|---|---|---|
| **3B contiguous 1024 B** | **+0.1019** | −0.0085 |
| pico (control) | +0.3487 | — |

This is the **first trustworthy positive result** on this asset. It confirms model-internally, with a
validated control and no cross-model step, that **contiguous 1024-byte rows carry real token data**,
consistent within a lane and not across it.

**What was ruled out for the cross-lane mapping**, all scored on 3000 cross-lane morphological pairs:

- **Linear-assignment fitting** (Hungarian on the outer-product cost matrix from cross-lane pairs):
  train DELTA +0.22–0.25 but **held-out test +0.003–0.008, identical to a shuffled-fit null**
  (+0.003–0.006). With ~140 constraints against a 2048-element permutation this overfits completely.
  Reported here because the train figure looks like success and is not.
- **~40 structured permutation families** — `(p + L·c) mod 2048` for c ∈ {1…512}, `p XOR (L≪k)` for
  k ∈ {0…10}, block rotations and in-block rotations for block sizes 8–256. Every one lands at
  −0.007 to −0.008, i.e. exactly the identity's −0.0087. No simple structured lane permutation works.
- **8-lane interleave extraction** (`pos = (t/8)(8D) + 8j + (t mod 8)`), the "mixture" hypothesis:
  **destroys** the same-lane signal (+0.102 → +0.005) while lifting cross-lane only to +0.003. Lane
  reversal and dim reversal behave identically. So the mixture model is excluded as well.

**Status.** Contiguous rows are confirmed as token data with a real 8-lane boundary. The cross-lane
dimension mapping is not a simple structured permutation, not recoverable by fitting from the
available constraint count, and not explained by an interleave. The honest position is that the
per-lane arrangement remains unknown, and that the next constraint should come from the ANE side
(what the `load_embeddings` op's gather actually does) rather than from more statistical search.
