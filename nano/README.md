# `afmplus-v11.0-nano` → GGUF

Decode procedure for Apple's **dense 3.33 B** on-device model and export to a GGUF that
`llama.cpp` loads and runs.

**No Apple weights are in this repository.** These scripts run against *your own device's* asset and
produce the GGUF locally.

## Result

The exported file **loads and runs** under `llama.cpp` b9430, exit 0.

| | |
|---|---|
| size | 6,718,421,888 B (6.257 GiB) |
| sha256 | `10604b66c53fe6c1bbfc81b9b9d8f92c55a2348041117788dab4b292792f70e9` |
| tensors | 618 (393 F16 2-D + 225 F32 norms) = `1 token_embd + 56x11 + 1 output_norm` |
| params | 3,355,688,960 |
| llama.cpp reads | `arch qwen3, n_layer 56, n_head 16, n_head_kv 2, n_embd_head_k/v 128, n_ff 6656, rope NEOX freq_base 500000, vocab SPM 262144, BOS 2, EOS 106` |
| speed | 4.0 t/s prompt, 2.9 t/s eval — **x86_64 binary under Rosetta, CPU only**; a native arm64 build will be far faster |

The chat template renders correctly and `<start_of_turn>` / `<end_of_turn>` tokenize as single
tokens, so the tokenizer metadata is right.

**It does not produce coherent text.** Four known-wrong components explain why — see *Known
incorrect*. This is published as a reproducible decode, not as a working model.

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

Three components are known to be wrong, and together they are why the output is incoherent:

1. **FFN channel ordering.** gate/up's output channels are ordered relative to down's input channels
   by an unrecovered permutation. This is the single remaining ordering unknown in the whole
   teardown; see `PICO_POSREAD_RESULT.md` §42.3, §54, §73. No instrument tried has power on it.
2. **RMSNorm gammas are shipped as all-ones.** nano's program declares **280 distinct
   `p_..._norm_weight` parameters**, so unlike the 300M sibling its gammas are *not* parameter-free.
   They have not been extracted, and the export substitutes ones.
3. **Sandwich post-norms are dropped.** nano applies **four** hidden RMSNorms per layer; the `qwen3`
   architecture expresses only the two pre-norms, so two per layer are lost. No GGUF architecture
   currently represents this.

And one unproven choice:

4. **`segment_1` K/V is duplicated.** Layers 35–55 have no K/V of their own and *which* layer they
   share from is not proven — a YOCO-style share from `segment_0` layer 34 is the best inference. The
   export byte-copies `blk.34`'s K/V into all of them (verified in the written file: `k34 == k55`
   exactly, while `q34 != q55`). This is **not a faithful export** of Apple's computation.

The file says so itself: `general.description` carries the full deviation notice, and custom
`afm.faithful=false`, `afm.warning`, `afm.ordering`, `afm.kv_sharing.export_hack` keys record it in
metadata. `llama.cpp` ignores them; `gguf-dump` shows them.

What *is* established, each against a null drawn from shuffling the data itself: the container and
2-bit palette, the scale maps (identity, every role), the block layout and role assignment, the
residual output ordering of the writers and input ordering of the readers, and that the attention
head layout is a gauge freedom rather than an unknown.
