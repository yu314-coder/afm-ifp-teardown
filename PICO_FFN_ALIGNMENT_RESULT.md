# pico forward: failure localized to the FFN neuron axis

Status as of 2026-07-19. All results below come from the 300M draft model
(`afmplus-v11.0-pico`, 24 dense layers, D=1024, 16Q/4KV heads, SwiGLU 3200, vocab 262144).

## 1. A real oracle, and the retraction of an earlier one

`pico_full.core` contains the model's **actual final logits** for the captured prompt:
262144 fp32 with only **144 masked** (`-40000`), i.e. ~262000 live tokens, `▁Paris` (id 9083)
argmax at 14.58. Scoring a candidate forward by **Pearson correlation against that whole vector**
is far stronger than ranking one token.

**This retracts a previously reported result.** The single-token oracle had reported the default
arrangement at `▁Paris` rank 686/37451 ("top 1.8%, real signal"). Under the correlation oracle that
same arrangement scores **corr −0.028** — pure noise. All four coarse arrangements score |corr| < 0.03:

| arrangement | corr vs true logits | ▁Paris rank |
|---|---|---|
| tileT=F grid=row | −0.0278 | 210528 |
| tileT=F grid=col | −0.0036 | 191499 |
| tileT=T grid=row | +0.0033 | 46788 |
| tileT=T grid=col | +0.0181 | 45428 |

A single token's rank over a 262k vocab is a high-variance statistic; it manufactured a
confident-looking false positive. It should not be used to pin an arrangement.

## 2. The model's real input (recovered from memory)

The `in_embeddings` buffer was located at ~`0x7a8000` (2048 B/row) by matching fp16 rows against
`round(row/SCALE)` int4 codes — the same fingerprint the embedding decode establishes. Decoding it
back through the vocab gives the exact token sequence:

```
▁the ▁capital ▁of ▁france ▁is <end_of_turn> \n <start_of_turn> model \n
510  5283      533  61038   567 110          111 109             4372  111
```

Two facts follow, both of which invalidate earlier forward runs: the harness **lowercases** the
prompt (`▁france` 61038, not `▁France` 7005), and the input carries a Gemma-style **chat template**,
so the model predicts from the final `\n`, not from "is". The same buffer holds a preceding
**safety-classifier turn** (`… 'paris' <end_of_turn> \n <start_of_turn> model \n 'safe'`), showing
afm runs a guard model before the LM.

## 3. Intermediate activations are not in host memory

Scanned all ~4M 2048-byte-aligned offsets of the 8.2 GB core for residual-stream buffers, using the
fingerprint that a hidden state `x = X0 + Δ` keeps each row correlated with its embedding row.
**No contiguous run of ≥4 correlated rows exists.** Every hit is an isolated row at corr ≈0.35,
which across ~40M comparisons is exactly the noise floor (128-dim random corr ≈0.09).

Only the two endpoints — input embeddings and final logits — cross into host memory. This is the
same ANE-locked boundary found on the 3B, and it closes off per-layer ground truth: the arrangement
cannot be verified one layer at a time.

## 4. The defect is one block, and it is the FFN

Correlation as a function of depth:

```
depth  0  (embed -> tied unembed, NO layers):  corr +0.0380   ▁Paris rank    2213
depth  1  (one transformer block):             corr -0.0397   ▁Paris rank  206862
depth 2..24:                                   never recovers
```

Depth 0 places ▁Paris at **2213 / 262000 (top 0.84%)** with positive correlation — independently
re-confirming that the embedding and the tied unembed are correct. A **single block** then destroys
it. This is *not* the 3B's gradual depth-accumulation failure.

Ablating that block (baseline = rank 2213):

| variant | ▁Paris rank |
|---|---|
| attention only, head reshape `(T,hd,n).transpose` | **4120 – 7355** (signal kept) |
| attention only, head reshape `(T,n,hd)` | 134251 – 218214 (destroyed) |
| **FFN only** | **238883** (destroyed) |

Two conclusions: the Q/K/V head layout is **dim-major**, not head-major; and the **FFN** is the
destroyer. The FFN's output rms is **2.53 against a 0.78 residual** — the delta is ~3× the stream it
is added to, and that magnitude is identical across all permutations (permutations cannot change
norms), so there is a scale error *in addition to* an ordering error.

## 5. An oracle-free alignment test, with a working positive control

The FFN neuron axis is a **gauge freedom**: any permutation applied consistently to gate columns, up
columns, and down rows leaves the function unchanged. So the "true" order is not needed — only
mutual agreement. gate/up column *j* and down row *j* are the same neuron, and in a trained network
neuron magnitude couples across in/out weights, giving a 3200-sample test that needs no ground truth.

**The control fires**, so the test has power:

```
corr(||gate_col_j||, ||up_col_j||)  = +0.341 (L0)  +0.214 (L23)     with tileT=False
                                    = +0.576 (L0)  +0.546 (L23)     with tileT=True
```

Tile transpose is not a column permutation — it changes which elements form each column — so this is
evidence that **gate/up want `tileT=True`**.

**down aligns with neither**, under every shape-valid assembly tried (12 down variants × block/tile
orders × row/col-major, plus gate/up assembly variants held against a fixed down):

```
corr(||down_row_j||, ||up_col_j||)   all |corr| <= 0.13, typically 0.02-0.08
random-permutation null, 99th pct     0.045 - 0.051
```

Nothing clears the null except one marginal L23 case (+0.090), unreplicated at L0 across 16 tests.

## 6. A geometry bug in the L (down) tile

Scale-group divisibility across the three tile classes:

```
N tile: 128 rows / 16 scales =  8.0   clean
s tile: 128 rows /  8 scales = 16.0   clean
L tile: 200 rows / 16 scales = 12.5   NOT clean   <- decoder used ceil() then truncated
```

A per-group scale that does not evenly tile its rows is a mis-parsed geometry. Reading the L tile as
**[256, 200]** gives 256/16 = **16 rows per group** — clean, and equal to the `s` class group size.
It also re-derives the down shape consistently (4 blocks × 256 = 1024 out, 16 tiles × 200 = 3200
neurons), meaning the tiles assemble into **downᵀ**. This geometry is more principled, but the
coupling test does **not** confirm it aligns (L0 shows nothing), so it is recorded as a corrected
parse, not a solved alignment.

## 7. Where this leaves the pico forward

Established and independently validated: embedding, tied unembed, weight *values*, attention head
layout, gate/up mutual consistency and tile orientation.

**Remaining blocker: the down-projection's neuron order.** Because coupling fails at noise for every
*coarse* (block/tile) permutation while the positive control succeeds, the misalignment is at a
granularity finer than tile permutation — i.e. **intra-tile element order**. That is the same class
of blocker the 3B's down-proj presented, which was only cracked bit-exactly using coreml2hwx ground
truth for the grouped-palettized z-order. Recovering pico's requires analogous ground truth; it is
not reachable by sweeping assemblies, and blind widening of the sweep is what produced the retracted
result in §1.
