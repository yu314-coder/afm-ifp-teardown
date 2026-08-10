# `afmplus-v11.0-nano` → GGUF

Decode procedure for Apple's **dense 3.33 B** on-device model and export to a GGUF that
`llama.cpp` loads and runs.

**No Apple weights are in this repository.** These scripts run against *your own device's* asset and
produce the GGUF locally.

## Result

The exported file **loads and runs**: 8.5 s load, ~2 tok/s on an M-series Mac, correct Gemma-style
chat template applied by the tokenizer metadata.

**It does not produce coherent text**, and two known-wrong components explain why — see *Known
incorrect* below. This is published as a reproducible decode, not as a working model.

```
The capital of France is
→ anyway anyway anyway anyway anyway anyway protected reigningdens def sheltered accessed
```

## The model

`UC_FM_LANGUAGE_INSTRUCT_3B_BASE_GENERIC_GENERIC_H16G_Cryptex.dmg` → `afmplus-v11.0-nano`.
It is the **dense** sibling of the sparse `instruct_3b`, so it has no experts, no
instruction-following pruning, and therefore none of the missing router metadata that blocks the
sparse model.

| | |
|---|---|
| layers | 56 = 35 `segment_0` + 21 `segment_1` |
| hidden / FFN / KV | 2048 / 6656 / 256 |
| heads | 16 Q × 128, 2 KV × 128, per-head QK-norm |
| parameters | 2.80 B in layers + 537 M tied embedding = **3.33 B** |
| weight codec | **2-bit** palette `[-1.5, -0.5, +0.5, +1.5]`, per-output fp16 scales |
| vocabulary | 262 144 |

`segment_1` layers carry **no K/V projections** — the last 21 layers reuse keys and values computed
earlier (cross-layer KV sharing), which Apple's own `manifest.plist` states.

## Container

```
tile = [palette 32 B][zeros 32 B] + N × ( [16 fp16 scales = 32 B][payload 16·cin/4 B] ) + [tail]
```

one scale per output, 16 outputs per sub-block, slot map `o = slot % 16`, `i = slot // 16`.

| class | stride | N | cin | tail | outputs |
|---|---|---|---|---|---|
| `s` | `0x2080` | 1 | 2048 | 32 | 16 |
| `a` | `0xa100` | 5 | 2048 | 32 | 80 |
| `e` | `0xe140` | 7 | 2048 | 32 | 112 |
| `d` | `0xd080` | 2 | 6656 | 0 | 32 |

Per-layer block units — the 966-block symbol-table sequence segments into these with **zero
remainder**:

```
segment_0 (18 blocks) sesseseeeaaeeedddd
  Q=(0s,1e) V=(2s) O=(3s,4e) K=(5s) gate=(6e,7e,8e,10a) up=(9a,11e,12e,13e) down=(14..17 d)
segment_1 (16 blocks) seeseeeaaeeedddd
  Q=(0s,1e) O=(2e,3s) gate/up/down as above, no K or V
```

## Scripts

| file | does |
|---|---|
| `decode_all.py` | all 350 weight tensors → npz, with a shape + trained-spectrum gate per tensor |
| `nano_embedding.py` | token embedding out of `program.odix` |
| `extract_nano_tokenizer.py` | the 262 144-token vocabulary |
| `seg1b.py` | derives the `segment_1` block layout |
| `nano_to_gguf.py` | writes the GGUF |
| `verify.py` | loads and runs it |

Run in that order. Requires numpy and `gguf-py` from llama.cpp.

## Known incorrect

Two components are known to be wrong, and both are why the output is incoherent:

1. **FFN channel ordering.** gate/up's output channels are ordered relative to down's input channels
   by an unrecovered permutation. This is the single remaining ordering unknown in the whole
   teardown; see `PICO_POSREAD_RESULT.md` §42.3, §54, §73. No instrument tried has power on it.
2. **RMSNorm gammas are shipped as all-ones.** nano's program declares **280 distinct
   `p_..._norm_weight` parameters**, so unlike the 300M sibling its gammas are *not* parameter-free.
   They have not been extracted, and the export substitutes ones.

And one unproven choice:

3. **`segment_1` K/V is duplicated.** Those layers have no K/V of their own and *which* layer they
   share from is not proven — a YOCO-style share from `segment_0` layer 34 is the best inference. The
   export duplicates K/V to produce a structurally valid 56-layer model. This is **not a faithful
   export** of Apple's computation.

What *is* established, each against a null drawn from shuffling the data itself: the container and
2-bit palette, the scale maps (identity, every role), the block layout and role assignment, the
residual output ordering of the writers and input ordering of the readers, and that the attention
head layout is a gauge freedom rather than an unknown.
