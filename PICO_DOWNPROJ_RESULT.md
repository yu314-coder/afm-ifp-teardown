# Pico down-projection decode — CRACKED

**Verdict: the 24 class-L down-projections now structure-test as real weights at mean true-SVD
R = 5.51 (min 4.65, max 6.90) over all 24 tensors; every magnitude/scramble control collapses to
R ≈ 1.0–1.2.** This clears the task bar (R > 3 real; scrambled ~1.4) decisively and to the *identical
standard* by which the 144 N-tensors (Q/K/V/O/gate/up, mean R = 4.0) and the 3B down-proj were
accepted. Reproduced independently by this adjudication pass, not merely reported.

Decoder: `/Volumes/D/fix/pico_shapes/pico_down_final_decode.py` → `decode_down(entry)` returns the
`[3200,1024]` float32 weight and prints the per-layer R table.

## Exact recipe

**Addressing.** `block_offsets` in `pico_weight_map.json` (role=='down') are **direct file offsets**
into `binary_0.hwx`. Each down tensor `[3200,1024]` = **4 L-blocks**; each L-block = **16 tiles** at
stride `0x6480` (25728 B) = 128 B header + 25600 B payload. 4·16·51200 int4 = 3 276 800 = 3200·1024.

**Codec (per tile).**
- payload = 25600 B = **51200 int4**, low-nibble-first: `byte → [b & 0xF, b >> 4]`.
- header `bytes[0:32]` = **16-entry fp16 per-tile codebook** (monotone, NF4/Lloyd-Max shape).
- **value = `codebook[nibble]`** — the per-tile palette. Beats linear `(nib − 7.5)` (R 5.51 vs 3.38),
  same palettized codec family as the accepted N-tensors.
- header `[32:64]` zero pad; `[64:96]` = 16 fp16 per-group scales (grouped-palettized scale slot,
  see caveat); `[96:128]` trailer.

**Geometry.** tile = **[200, 256]** row-major. block `b` → columns `[b·256 : (b+1)·256]`;
tile `t` → rows `[t·200 : (t+1)·200]` → `[3200, 1024]`.

## Why this is a genuine crack (not a magnitude artifact) — controls, true SVD, 8 tensors

| control (codebook decode, tile [200,256]) | mean R | meaning |
|---|---|---|
| **real** | **5.42** | passes R > 3 |
| intra-tile shuffle (preserve exact per-tile value multiset, destroy order) | **1.00** | z-order carries the structure |
| random nibbles through same per-tile codebook | **1.20** | not a per-tile codebook-range artifact |
| per-tile RMS normalization (remove all per-tile magnitude) | **5.45** | **not a magnitude artifact** |
| linear `(nib−7.5)` decode | 3.38 | codebook is the correct dequant |

The three magnitude controls are the decisive test. `intra-tile shuffle → 1.00` and
`per-tile-normalized → 5.45` together prove the R elevation is carried by the **within-tile order of
the actual nibble values**, not by any per-tile/per-channel magnitude banding. This **refutes** the
concern that codebook-decode R is a magnitude floor (~3.67 was claimed by a sibling analysis; the
measured floor is ~1.2), and it **rejects** the `reshape(16,3200) × per-channel-scale` recipe, whose
R = 11–19 is a pure per-channel-magnitude inflation (random nibbles there also score ~19).

## Honest limits — the residual convention floor (identical to 3B and to the accepted N-tensors)

R (any singular-value metric) confirms the **codec + tile geometry + contiguity + block placement**,
but **cannot** arbitrate these spectrum-invariant relabelings, so they are NOT pinned by this test:

- **Output-column relabel**: a globally-consistent permutation of the 256 output columns (row-major
  vs 3B-OCG interleave) is spectrum-invariant — R is identical. Row-major is used for concreteness.
- **Aspect ratio** among row-major reshapes: [200,256]=5.15, [100,512]=5.50, [50,1024]=4.84,
  [400,128]=4.05, [256,200]=4.79 all pass. [200,256] is chosen by the AFM-3B 256-wide-tile convention
  and the natural 4-block×256 geometry, not by R.
- **Block placement**: block→col (5.15) slightly beats block→row (4.74); both pass.
- **Per-group scale slot** `[64:96]`: applying it is spectrum-neutral, so R cannot confirm it. Left
  out of the validated value formula (its use elsewhere produced only magnitude inflation).

These are exactly the ANE hardware-tiling convention residuals documented for the 3B down-proj
(`afm_odix/downproj_decoded.json`) and are below the structure test's resolution — the same floor at
which the Q/K/V/O/gate/up tensors were already accepted. Byte-exact micro-permutation would require a
forward oracle-rank fit (the 3B route), not the SV structure test.

## Artifacts
- `/Volumes/D/fix/pico_shapes/pico_down_final_decode.py` — `decode_down(entry)` + all-24 R table
- `/Volumes/D/fix/pico_shapes/adjudicate.py` — magnitude-artifact controls (true SVD)
- `/Volumes/D/fix/pico_shapes/adjudicate2.py` — aspect-ratio / placement / separability characterization
