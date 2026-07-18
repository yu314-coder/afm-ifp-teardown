# Pico (300M) token embedding in `program.odix` — VERDICT: **CRACKED**

Adjudicated independently; every load-bearing claim below was re-verified against the
real file, not taken from any agent's self-report.

Target:
`/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/031c7be6f8fddbff0a6650fee75e345b1ee9613c.asset/.AssetData/model.odixpackage/program.odix`
(136,052,680 bytes)

Decoder: `/Volumes/D/fix/pico_shapes/pico_embedding_decode.py`

---

## Verdict

**CRACKED.** A decoder reproduces **5 of 5** captured ground-truth rows **bit-exactly**
(`np.array_equal` on float32, not merely cosine ~1) directly from token ids.

```
key      token_id cosine       bit-exact
4803     4803     1.00000006   True
12418    12418    1.00000002   True
9619     9619     1.00000007   True
44862    44862    1.00000009   True
5864     43534    1.00000004   True     <- see "mislabelled key" below

ALL 5 BIT-EXACT: True
```

The falsification agent's hypothesis (embedding absent from `program.odix`,
ANE-baked or generated at load) is **refuted**: the table is present in
`program.odix` and is now fully decoded.

---

## The scheme

Per-**channel** symmetric int4, zero point 0:

```
value[t, d] = float16( q[t, d] ) * scale[d]
```

- `q` — signed 4-bit two's-complement nibble, `q ∈ [-8, 7]`
- `scale` — **1024 fp16, one per embedding dimension, shared by all 262,144 rows**

Not a codebook. This is why the "FFN-palettized codebook" and
"[262144 fp16 scale table] + int4" families all scored at chance: the scale is
per-column, not per-row and not per-group-along-the-row.

**Independently re-verified from ground truth**: dividing each captured row by the
scale vector read out of the file yields integers — `max|q| = 5.001`, mean residual
`5e-06 … 2.7e-05`, and 100% / 99.8% / 99.6% of entries within 1e-3 of an integer.
That is an assumption-free confirmation, since the scale came from the file and the
values came from the live model.

### Scale table

**File byte 135,872,328 (`0x8193F48`)**, 1024 × fp16, range 0.014114 … 0.151200, no zeros.

Verified stored as **88 identical 2048-byte replicas in three runs**:

| offset | hex | count |
|---|---|---|
| 135,872,328 | `0x8193F48` | ×64 |
| 136,003,464 | `0x81B3F88` | ×16 |
| 136,036,296 | `0x81BBFC8` | ×8 |

The **64 / 16 / 8** split matches the op names
`gather_embeddings_{64,16,8}` exactly — independent corroboration that this is the
embedding's scale tensor.

---

## The layout (what defeated 8 prior sweeps)

```
block = 8 tokens x 1024 dims = 8192 nibbles = 4096 bytes
lane  = t % 8
SKEW  = [0, 0, 0, 0, 8184, 8184, 8184, 8184]

nibble_index(t, d) = 1203476 + (t // 8) * 8192 + 8 * d + lane + SKEW[lane]
byte = nibble_index >> 1 ;  low nibble if nibble_index is even
```

Payload starts at byte **601,738** — *before* the declared data-section start
`0x93ec8` = 605,896 — and spans exactly `262144 * 512 = 134,217,728` bytes.

Two properties make this adversarial to blind sweeping:

1. **Element-wise, not row-wise, interleave.** 8 vocab rows are interleaved at
   *nibble* granularity with dim-stride 8. A token's 1024 codes are therefore never
   contiguous anywhere in the file, so every contiguous-row and 512-byte-window
   search correctly returned zero hits.
2. **The lane skew.** Lanes 4–7 sit 8184 nibbles later than lanes 0–3. Omit it and
   lanes 0–3 decode *perfectly* while lanes 4–7 decode to noise — half the table
   looks right and aggregate statistics say the base is correct everywhere.

---

## The lane trap — caught here, and it changes the ranking

The 5 captured rows only cover lanes `t%8 ∈ {2, 3, 6}`. **Five of eight lanes are
not pinned by the ground truth at all.** A decoder can be bit-exact on all 5 captured
rows and still silently corrupt 25% of the vocabulary.

I tested this rather than assuming it. Using 17,272 case-variant pairs
(`▁word` / `▁Word`) from `tok_vocab.json` as an independent probe:

- **With the skew** (verified formula): per-lane mean cosine
  `0.6497 0.6519 0.6494 0.6495 0.6484 0.6481 0.6501 0.6498` — **flat across all 8
  lanes**; full 8×8 residue matrix uniform at 0.639–0.662; overall real 0.6496 vs
  shuffled control 0.2348.
- **Without it** (the `t+2` formulation): lanes 6 and 7 collapse to **0.0610 / 0.0613**,
  reading cos ≈ −0.01 against lanes 0–5 but ≈ 0.28 against each other — a consistent
  displacement, not corruption.

This was **not** visible on the 5 ground-truth rows, which pass bit-exactly under both
formulations. It is only visible on the full table.

### Additional whole-table checks

- **140 zero rows, perfectly contiguous at ids 262,004–262,143** (unused vocab tail).
- Nearest neighbours coherent, including in the previously-broken lanes
  (`▁nodded` is lane 7, `每天` is lane 6):

| probe | lane | nearest neighbours |
|---|---|---|
| `▁dog` | 3 | ▁Dog .73, Dog .69, dog .65, ▁dogs .65, 狗 .56 |
| `▁king` | 3 | ▁King .71, King .68, ▁kings .66, ▁queen .59, 国王 .54 |
| `▁Paris` | 3 | Paris .77, ▁París .63, ▁Parisian .62, 巴黎 .60 |
| `▁coffee` | 5 | ▁Coffee .76, coffee .72, 咖啡 .64, コーヒー .62, ▁커피 .61 |
| `▁France` | 5 | France .77, ▁Frankreich .64, 法国 .63, フランス .62 |
| `每天` | **6** | 每日 .70, 毎日 .63, Daily .61, ▁täglich .59 |
| `▁nodded` | **7** | ▁nodding .66, ▁nods .65, ▁smiled .56, 点头 .53 |

---

## Ground-truth file correction

**`captured_embeddings.npz` key `'5864'` is mislabelled.** That vector is bit-exactly
the row of token **43534 = `▁london`** (lowercase), not 5864 = `▁London`.

Verified against `/Volumes/D/fix/afm_odix/tok_vocab.json`, which aligns at offset 0
(`v[4803]='▁dog'`, `v[12418]='▁dogs'`, `v[9619]='▁king'`, `v[44862]='▁kings'`,
`v[43534]='▁london'`, `v[5864]='▁London'`). Token 5864's actual row has cosine
**0.646** to the stored vector. The other four keys are correct.

Worth checking the capture harness for a lowercasing step.

---

## Adjudication of the five submissions

All four "CRACKED" submissions converge on the same scheme and the same scale offset,
and three of them are algebraically the same address law. They are **not** equally
correct.

| # | claim | formula correct? | artifact correct? |
|---|---|---|---|
| 2 | CRACKED | **yes** (skew form) | `pico_embed_codes_int8.npy` — **0 / 262,144 rows differ** |
| 3 | CRACKED | **NO** — wrong on lanes `t%8 ∈ {4,5}` | `pico_embedding_fp16.npy` — **65,502 rows wrong** |
| 4 | CRACKED | **yes** (equivalent to #2) | `pico_emb_codes_int4.npy` — 0 rows differ at documented `off=-4` |
| 5 | CRACKED | **yes** (equivalent to #2) | `pico_embedding_table.npy` — **0 / 262,144 rows differ** |
| 1 | PARTIAL | scheme + scale correct; addressing unsolved | `embedding_scale_vector.npy` — scale correct |

Notes on the adjudication:

- **#2, #4, #5 are byte-identical address laws** despite three different-looking
  formulations. Verified by direct comparison of generated addresses.
- **#3 is the odd one out.** Its `r = t+2` reformulation differs from the others by
  exactly `+8184` on `t%8 ∈ {4,5}`, i.e. it drops the skew for two lanes. It passes
  bit-exactness on all 5 ground-truth rows because none of them land in those lanes.
  Its dumped table is wrong for 65,502 tokens, all in lanes 4 and 5. **Do not use
  `pico_embedding_fp16.npy`.**
- **#5 diagnosed the lane trap correctly but overstated its own fix.** Its writeup
  says lanes 6/7 read cos −0.013 against the rest — that is a real and correctly
  characterized failure mode, and its final published `D = [0,1,2,3,8188,8189,8190,8191]`
  is in fact the *correct* law. Its own table is clean.
- **#1's negative results are sound and were not overturned** — contiguous-row,
  512-byte-window, column-major, and vblock/dblock families really are all dead. Its
  error was concluding the search space was exhausted; the element-wise nibble
  interleave was outside every family it enumerated. Its scale-tensor localization was
  correct and is the piece all the successful decodes depend on.
- **#3's claim that `pico_embedding_table.npy` "is not mine and has different
  contents"** is explained: that file is #5's, and it is the correct one; #3's differs
  because #3's is wrong.
- Claims of an "88 × 1024" scale region are right in total but the structure is
  three runs of 64/16/8, which is the more informative fact.

### Artifacts safe to use

- `/Volumes/D/fix/pico_shapes/pico_embedding_decode.py` (this adjudication's decoder)
- `/Volumes/D/fix/pico_shapes/pico_embedding_table.npy` — `[262144,1024]` fp16, **direct token indexing**
- `/Volumes/D/fix/pico_shapes/pico_embed_codes_int8.npy` — `[262144,1024]` int8 codes, direct indexing
- any of the five `*scale*.npy` files (all verified equal to the file's scale vector)

### Artifacts to delete or fix

- `/Volumes/D/fix/pico_shapes/pico_embedding_fp16.npy` — **wrong for 65,502 tokens**
- `/Volumes/D/fix/pico_shapes/pico_emb_codes_int4.npy` — correct but uses a `t-4` row
  offset; only safe if that convention is honoured (and it cannot represent tokens 0–3)

---

## Consequence

The embedding is **tied** to the unembed, so

```
logits = h @ table.T
```

This makes the captured ground-truth logits in `/Volumes/D/fix/pico_oracle/` usable as
a functional oracle. That end-to-end loop was **not** exercised here — closing it still
requires the final hidden state out of `pico_full.core`, which this adjudication did not
touch. Treat the oracle as unblocked, not as demonstrated.
