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
