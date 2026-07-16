# ANE Weight-Encoding Decode — Plan of Record

**Goal.** Recover the AFM token embedding (and, as a by-product, validate/resolve the FFN
expert-weight shapes) by cracking the Apple Neural Engine constant-encoding used in
`binary_0.hwx`, rather than by searching offsets/layouts in the shipped weight files — which is now
an exhausted, four-front negative.

This document exists so the next effort starts from established facts. Every "Known" below was
pinned experimentally in the 2026-07-15 pass; see `paper/afm_teardown.tex §find:embed`,
`ODIX_DECOMPILER.md`, and the `afm-embedding-not-validated` project note.

---

## 0. Why this is the only remaining path

The token embedding's **data** is absent from every plain-table location:

| Surface | Test | Result |
|---|---|---|
| Raster `ifp_rasterized_weights.bin` (4.9 GB) | ~4.16 M `[V,D]` offsets + 72 K `[D,V]` transposed, 4-bit & int8, tiled & contiguous, scaled & unscaled | best +0.089 |
| `main-h16g.odix` blob | the one vocab-structured table (pinned +0.777/0.78) | +0.003 → it is `c_sparsity_embedding`, **not** the embedding |
| `binary_0.hwx` on-disk sections | `[V,D]` token-row read (defeated by ANE swizzle) | inconclusive — **see below** |
| odix op-graph | `gather_embeddings` → 512 chunks of `[512,1536]` 4-bit, dtype `0x40012` | chunks are **ANE-resident** |

**The hwx is the leading location, not a dead end** (corrected 2026-07-15): `__KERN_0` is **133.73 MB**
@ fileoff 104,660,992, entropy **7.77**; `__KERN_1` 25.32 MB, entropy 7.94; `__TEXT.__const`
46.65 MB, entropy 4.77. `sum(__KERN_* + __const) = 437.9 MB > 201.3 MB`, so a full 4-bit embedding
**fits on disk**. The high entropy is the signature of ANE-encoded/palettized weights. The earlier
"no section ≥ 201 MB" note was a section-bounds bug; the earlier "no vocab structure" test assumed a
`[V,D]` token-row layout, which the ANE tiling defeats — so it does **not** rule the hwx out.

The scoring instrument is trustworthy: a real 4-bit LM embedding (Qwen3-4B `token_embd`,
re-quantized to this model's exact signed-4-bit + per-token-scale format) scores **+0.533** on it;
4-bit costs ~2%. Nothing in the shipped files exceeds +0.089. So the embedding is **ANE-baked** —
present only inside the compiled ANE program in ANE constant-encoding, or patched into the runtime
`__MKERN`/`__bss` regions from a raster slice under an ANE byte-permutation that defeats a
token-row read.

**The one method that produces ground truth is `coreml2hwx`**: compile a *known* weight through
Apple's own ANE toolchain, read back the exact bytes, and learn the encoding by comparison. This is
the same method that validated the dense 8×128 de-swizzle (real-weight decay 8–11 vs scrambled
1.45). It sidesteps the fundamental problem — no ground-truth activations — because *we* supply the
input.

---

## 1. Established facts to build on (do not re-derive)

- **Embedding logical shape:** `[V=262144, D=1536]`, 4-bit, per-token scale. VOCAB confirmed four
  independent ways; `152064` (used historically) is Qwen's size, wrong.
- **Storage granularity:** 512 chunks × `[512 tokens, 1536]`, **393216 bytes/chunk**, dtype
  `0x40012` (low byte 18 = the 4-bit-palettized type shared with FFN constants). `512×393216 =
  201,326,592 = V·D/2` exactly. Gathered by `gather_embeddings_{8,16,64}` (batch, not chunk, size).
- **Dequant:** signed two's-complement 4-bit (`q = n−16 if n≥8 else n`), **not** `idx−7.5`. Nibble
  histogram of trained rows peaks at 0 and 15, empty at 7/8.
- **Codebook/scale for FFN weights:** 4-bit index → 16-entry codebook `afm_codebook_deswz.npy` →
  per-1024 fp16 block scale → ANE 8×128 tile de-swizzle
  `reshape(Co//8,Ci//128,128,8).transpose(0,3,1,2).reshape(Co,Ci)`. The embedding may use a
  *different* tiling (it is a gather target, not a conv weight) — that is the unknown to crack.
- **`binary_0.hwx` layout (Mach-O, magic `0xbeefface`, `LC_SEGMENT_64`), on-disk data sections:**
  `__KERN_0` **133.73 MB** @ 104,660,992 (entropy 7.77), `__TEXT.__const` **46.65 MB** @ 57,999,360
  (entropy 4.77), `__KERN_1` **25.32 MB** @ 238,387,200 (entropy 7.94), plus `__TEXT.__text` 53.37 MB
  (code). Runtime-allocated/0-on-disk: `__MKERN_0` 202.15 MB (= ifp1 expert-coeff kernel per
  `afm-hwx-expert-dma`), `__MKERN_22` 30.08 MB, `__DATA.__bss` 84 MB. The `__MKERN` regions are
  runtime-**patched** from the raster via a content-hash `dsid` (see `afm_odix/hwx_expert_dma.json`).
  `__KERN_0`+`__const`+`__KERN_1` = 205.7 MB of high-entropy on-disk data ≥ the 201.3 MB embedding —
  `__KERN_0` (133.73 MB, entropy 7.77) is the **prime target region**.
- **Toolchain:** `coreml2hwx` at `/Volumes/D/fix/coreml_to_ane_hwx/coreml2hwx`; build harness
  `disasm/pbuild.py`, readback `disasm/posread_full.py` / `dense_posread.py`; env
  `/Volumes/D/fix/anevenv/bin/python3` (coremltools 9.0) to compile, `/usr/local/bin/python3`
  (torch/numpy) to analyze.
- **Validation oracle:** the semantic probe (`/tmp/control.py`, `/tmp/gguf.py`). Target ≥ **+0.30**
  (a correct decode); anything ≤ +0.10 is noise. Always pair with the **shift control** (offset the
  table by whole tiles; a real decode collapses off-target) and the **id-matched** random baseline
  (low-id tokens have inflated mutual cosine — a band artifact that faked results twice).

---

## 2. Phase 0 — Decide the theatre: raster-swizzled vs hwx-baked (1–2 days)

Before decoding anything, settle *where* the 201 MB of embedding data physically lives. Two
mutually exclusive hypotheses, one cheap discriminator each.

- **H-raster:** the embedding is in `ifp_rasterized_weights.bin` under an ANE byte-permutation the
  token-row sweeps couldn't invert. *Discriminator:* the zero-rows of the ~6000 contiguous untrained
  `<unused>` tokens must survive as a large zero region under *any* within-tile permutation (zeros
  permute to zeros). The raster zero-scan found none ⇒ **H-raster is unlikely** unless the ANE tiling
  interleaves the token axis across the whole vocab (mixing unused with trained). Test that directly:
  compile a `[512,1536]` constant whose rows 400–511 are zero through `coreml2hwx`, read back, and
  see whether the zero rows stay contiguous or scatter. If they scatter vocab-wide, re-scan the
  raster for the *scattered* zero fingerprint.
- **H-hwx (leading):** the embedding is compiled into `binary_0.hwx`. `__KERN_0` alone (133.73 MB,
  entropy 7.77) plus `__const`/`__KERN_1` gives 205.7 MB of high-entropy on-disk data ≥ the 201.3 MB
  needed, so a full 4-bit embedding **fits on disk**. *Discriminator:* once Phase 1 yields the ANE
  codec, decode `__KERN_0` at chunk stride and score with the oracle. *Pre-Phase-1 cheap check:*
  compile a `[512,1536]` constant with a known zero-block through `coreml2hwx`, learn where zeros
  land, then look for that fingerprint in `__KERN_0`. Also trace the `gather_embeddings` `dsid`
  through `hwx_expert_dma.json` — it should resolve to a `__KERN`/`__MKERN` region, pinning the base.

**Exit:** a single sentence — "the embedding bytes are at `<file:offset>` under encoding `<name>`."
Everything after depends on this. Current best guess: **`__KERN_0` @ 104,660,992 in `binary_0.hwx`,
ANE-palettized, 512 chunks of 393216 B.**

### Phase 0 findings (2026-07-15, in progress)

Ran the discriminators. Results — encoding largely clarified, exact data location still open:

- **ENCODING CONFIRMED = `PalettizedConv2D` + `lut_to_dense`.** `specialized_model_0.mpsgraph`
  names the layer weights as `PalettizedConv2D_{799,740,809,860,…}` with `lut_to_dense` ops —
  i.e. exactly the 4-bit-index → 16-entry codebook codec already cracked for FFN weights. The
  token embedding / logit projection is the same family. So there is **no new codec to learn**,
  only the tiling.
- **THE ANE TILING IS EXPLICIT AS AFFINE MAPS in the mpsgraph** — invertible directly, likely
  removing the need for a `coreml2hwx` round-trip to *learn* the layout. The 8-way-tiled maps are:
  `(d0,d1,d2,d3,d4) -> d0*9175040 + d1*262144 + (d2//8)*2048 + d3*2048 + d4*8 + d2%8` and
  `(d0,d1,d2,d3) -> d0*504102912 + (d1//8)*12288 + d2*12288 + d3*8 + d1%8` (note `12288 = 8×1536`).
  A separate family `d0*K + d1*32 + d2*32 + d3`, K ∈ {16384,32768,49152,65536,131072,262144},
  factors as **32 × {512,1024,1536,2048,4096,8192}** (49152 = 32×1536 = 32×hidden).
- **⚠️ CAUTION — `262144` in the maps is NOT confirmed as vocab.** It equals both `V` and
  `32×8192` (32 × max-context). No vocab-sized `tensor<>` shape or `logit`/`token_embedding` op
  name was findable in the binary MIL. Do not assume `262144 = vocab` without shape confirmation
  (this session burned six such assumptions). Resolve in Phase 1 by reading the op's I/O shapes.
- **hwx section discriminator (see §0 table):** `__KERN_0`/`__KERN_1` are uniform entropy 7.8–7.9
  with **no unused-token zero tail** at any 393216-chunk boundary. Two readings, both live: (a) they
  are affine-**tiled** palettized weights (tiling scatters the unused-token zeros vocab-wide, so no
  contiguous tail appears — *consistent* with an embedding container), or (b) they are not the
  embedding. `__KERN_0` is 133.73 MB < 201 MB, so it cannot hold the full 4-bit embedding alone.
- **NEW leading candidate: `__MKERN_0` = 202,145,792 B ≈ the 201,326,592 B embedding** (runtime-
  allocated, 0-on-disk, patched from the raster at load). If `__MKERN_0` is the resident embedding
  buffer, the **raster holds the source** — but affine-tiled, which is exactly why the `[V,D]`
  token-row raster sweeps missed it. This is the most likely resolution and the Phase-1 target:
  trace the `gather`/patch `dsid` for `__MKERN_0` to its raster region, then invert the affine map
  on that region.

**Phase 0 status:** encoding solved; location narrowed to {`__KERN_0` on-disk, affine-tiled} vs
{raster-source patched into `__MKERN_0`, affine-tiled} — decide by tracing the `__MKERN_0` patch
`dsid` (extend `afm_odix/hwx_expert_dma.json`, which covered only the expert `__MKERN`s) and by
confirming the `262144` dimension is vocab.

---

## 3. Phase 1 — Learn the encoding by `coreml2hwx` round-trip (1–2 weeks)

Compile controlled constants and read back the exact ANE bytes; fit the transform.

1. **Positional-probe constant.** Build a `[512, 1536]` constant with value `[r,c] = r*1536 + c`
   (a unique integer per element), dtype matched to `0x40012`, and compile it via `pbuild.py`
   through `coreml2hwx`. Read back the emitted section bytes (`posread_full.py`). Each output byte
   position now maps to a known `(r,c)` → **the exact spatial permutation**, directly, no fitting.
   This is how the dense 8×128 map was recovered; reuse `recover_swizzle()` in `dense_posread.py`.
2. **Palettization/codebook.** Compile a constant with a known real-valued distribution and read
   back the 4-bit indices + scale layout. Confirm: codebook identity (is it
   `afm_codebook_deswz.npy` or a distinct embedding codebook?), scale granularity (per-token, per
   block, or per-tile), and zero-point (expected: two's-complement, scalar).
3. **Chunk boundary.** Compile two adjacent `[512,1536]` chunks and confirm the 393216-byte stride
   and whether chunks are independently tiled (they should be — the gather addresses per chunk).
4. **dtype `0x40012` semantics.** The low byte is 18; enumerate what the toolchain emits for
   `0x40012` vs neighbouring dtypes to nail element width and signedness from the tool, not
   inference.

**Deliverable:** `ane_embed_codec.py` implementing `encode(W)->bytes` and `decode(bytes)->W` for a
`[512,1536]` dtype-`0x40012` chunk, verified bit-exact against the round-trip.

**Phase 1 status (2026-07-15): ENGINE VERIFIED.** A minimal palettized 4-bit 1×1 conv
(`Cin=256,Cout=128`) compiled through `coreml2hwx` (rc=0, coremltools 9.0 in `anevenv`) to a hwx
with **magic `cefaefbe` and sections `__kern_0`(17408 B) + `__const`(16384 B = LUT)** — the *same*
format as the shipped `binary_0.hwx`. So the toolchain reproduces Apple's encoding, and the
`posread_full.py` positional-read (which cracked the down-proj 48×256 z-order) applies directly.
**Remaining Phase-1 work** (multi-session): (a) build a positional-probe conv at the *embedding's*
shape — blocked on confirming that shape, since `262144` appears only 2× in the mpsgraph (vs `8192`
12×, `1536` 5×) and is likely `32×8192` not vocab, so the embedding op's Cout must be read from its
I/O binding or found by compiling candidate shapes and matching the shipped section sizes; (b) read
back the tiling; (c) fold in the per-token scale. Smoke test: the `if __name__` block in
`disasm/pbuild.py` already sweeps G∈{1,4,8,23} and prints `opb/OCG/CoeffSize`.

**Kill criterion:** if `coreml2hwx` refuses the embedding dtype/shape or emits a fundamentally
different structure than the gather target in `binary_0.hwx` (compare byte histograms / section
structure), stop — the shipped encoding is not reproducible by the public toolchain, and the wall
is hard. Record the divergence.

---

## 4. Phase 2 — Invert on the real data + validate (2–4 days)

1. Apply `ane_embed_codec.decode` to the real embedding bytes located in Phase 0, per chunk, to
   reconstruct `[262144, 1536]`.
2. Score with the semantic oracle. **Success = orthographic-variant gap ≥ +0.30, and the shift
   control collapses off-target.** Spot-check nearest neighbours: `▁Paris → ▁London/▁Berlin/▁Rome`
   at cosine ≳ 0.4.
3. If gap is 0.1–0.3 (partial): the spatial map is close but a secondary axis (per-token scale
   assignment, chunk order, sign) is off — sweep those *few* discrete options against the oracle
   (now safe: the oracle is calibrated and the search is tiny).

**Deliverable:** `emb.pt` `[262144,1536]` fp32 that passes the probe; the *first* recovered AFM
embedding, and the first non-vacuous validation of any AFM tensor against external semantics.

---

## 5. Phase 3 — Generalize to FFN constants (stretch, unblocks the forward)

The same `coreml2hwx` map, re-parametrized per shape, resolves the outstanding forward gaps:

- **Per-constant expert width** (the `219`-vs-`235` gap, and the "variable width 42–232" the odix
  value-table exposed): compile constants at candidate `[C_out, 1536]` widths, confirm the tiling,
  then read exact per-layer `C_out` from the odix value-table `f0/f4` shape vectors once the
  type-table indirection (`type_index → type table → symbol pool`; see `ODIX_DECOMPILER.md`) is
  resolved — the type table is nested in a module op / value sub-table, not a top-level vector.
- **Output norm + layer pairing:** the remaining two calibration pieces for a full forward, both
  read from the same op-graph once shapes are exact.

Only attempt Phase 3 if Phase 2 succeeds; otherwise the forward stays walled regardless.

---

## 6. Effort, risk, and decision points

- **Realistic effort:** Phase 0 ≈ 1–2 days; Phase 1 ≈ 1–2 weeks (the real work — ANE tiling for a
  gather target is not the conv 8×128 map); Phase 2 ≈ days; Phase 3 ≈ 1–2 weeks. Total: a focused
  multi-week effort, single-threaded on `coreml2hwx` round-trips.
- **Central risk:** the shipped ANE encoding may use compile-time state (fusion, kernel
  coalescing — note `--optimize-kernel-coalescing` / `--disable-kernel-streaming` strings in the
  hwx) that `coreml2hwx` does not reproduce for a standalone constant. If so, the round-trip map
  won't match the baked bytes and the wall is genuinely hard. Phase 1's kill criterion catches this
  early.
- **What NOT to do:** no more offset/layout sweeps of the shipped files against the oracle — that
  space is exhausted (4.16 M raster candidates, all four surfaces). Statistical shortcuts without
  ground truth produced six falsified-by-control mirages in one session; the `coreml2hwx`
  round-trip is the only method here that yields facts.

---

## 7. Standing invariants for whoever picks this up

1. **Never claim a decode without the shift control** (offset by whole tiles → gap must collapse)
   **and** the id-matched random baseline. Raw cosine and self-ranking are both vacuous here.
2. **Calibrate every probe on a real model at the same quantization** before trusting a number.
3. **Never u32-scan a FlatBuffer for sizes** — its vtables are u16 (this caused a retracted
   `alloc_const` claim). Use `src/odix_fb.py`.
4. **Publish only original research** — decode code and findings, never Apple's weights/tokenizer
   data (`.gitignore` blocks `*.pt *.bin *.hwx *.odix *.npy …`).
