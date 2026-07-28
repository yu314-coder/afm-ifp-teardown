# pico (300M) → GGUF export

`tools/pico_to_gguf.py` exports `afmplus-v11.0-pico` to a GGUF that **llama.cpp loads and runs**.
The `.gguf` itself is Apple's weights and tokenizer, so it is `.gitignore`d and never published —
only the exporter is in this repo.

## What it produces

```
afmplus-v11.0-pico-F16.gguf     1.14 GB, 218 tensors, 18 metadata keys
```

Mapped onto the `llama` architecture, which fits pico exactly (RMSNorm, SwiGLU, GQA, RoPE):

| GGUF key | value |
|---|---|
| `llama.block_count` | 24 |
| `llama.embedding_length` | 1024 |
| `llama.feed_forward_length` | 3200 |
| `llama.attention.head_count` / `_kv` | 16 / 4 |
| `llama.rope.freq_base` | 500000.0 |
| `llama.rope.dimension_count` | 64 |
| `tokenizer.ggml.model` | llama (SPM), 262144 tokens |
| `tokenizer.ggml.eos_token_id` | 110 = `<end_of_turn>` (recovered from `in_embeddings`) |

Tensors are F16: `token_embd.weight` from the bit-exact embedding decoder, per-layer
`attn_{q,k,v,output}` and `ffn_{gate,up,down}` from the validated weight decoder, and `attn_norm` /
`ffn_norm` / `output_norm` written as **ones** (γ is folded into the adjacent linear at ANE compile,
so unit norms are correct rather than a placeholder).

Two portability details worth recording: llama.cpp asserts
`id_to_token.size() == token_to_id.size()`, so the four duplicated specials in pico's vocab
(`<unk>`, `<bos>`, `<eos>`, `<pad>` are mirrored at ids 4–7) must be made unique or the model aborts
on load; and GGUF wants `ne[0] = input dim`, so each decoded `[cin, cout]` matrix is written
transposed with `dims = [cin, cout]`.

## Honest status of the output

It loads and generates at ~15 tok/s on this machine. **The generated text is incoherent**, e.g.

```
$ llama-completion -m afmplus-v11.0-pico-F16.gguf -p "the capital of france is" -n 12 --temp 0
 the capital of france ishamdul placés philanthdisturbance कैरेट IEnumerator娛sembles輯classedGetAxis
```

This is the expected consequence of the open blocker in `PICO_POSREAD_RESULT.md` §11–12: the
per-layer weights are decoded with a z-order that is round-trip-validated for the compilable
`OutTrans=0` mode, while the shipped tiles are `OutTrans=1`, whose coefficient ordering cannot be
read with available tooling. The embedding, tokenizer, architecture metadata and file structure are
all correct — the layer weight *ordering* is not.

So the export is best understood as a **correct container around partially-ordered weights**: useful
as a harness (it makes the reconstruction runnable under a standard engine, and any future fix to
the ordering drops straight in), not as a working language model.

## Verified artifact (final)

```
file    afmplus-v11.0-pico-F16.gguf
size    1,141,283,072 bytes (1.1 GB)
sha256  fd9c04151b4c8bb125d0771390953a62046b12371a5291f68af81fc1cbf39c76
format  GGUF v3, 218 tensors, 18 metadata keys, file_type 1 (F16)
arch    llama | 24 blocks | D 1024 | FFN 3200 | 16 Q heads / 4 KV heads
        RoPE theta 500000 | ctx 4096 | vocab 262144
```

Re-verified end-to-end under `llama.cpp` (`llama-completion`): loads clean, ~2.2 GB resident,
**15.8 tok/s**, output still incoherent exactly as described above. Only 5 of 7 weight roles per
layer (`Q`, `K`, `V`, `gate`, `up`) are correctly ordered; the two residual-writing roles (`O`,
`down`) carry the unresolved `OutTrans=1` order.

Scope of the remaining defect, quantified from Apple's own binary (`PICO_POSREAD_RESULT.md` §18):
of pico's 988 weight-bearing conv tasks, **180 are `OutTrans=1`** and 808 are `OutTrans=0`. So
roughly **82% of the weight-bearing tasks decode with a round-trip-validated order** and 18% do not.
That fraction, not the file, is what stands between this artifact and coherent text.

**Do not** substitute one of the speculative `O`/`down` orderings from §13 (`ifast`, `obank`,
`ifast_obank`) to make the output look better -- all of them remain inside the measured noise band,
so shipping one would present an unvalidated guess as a solve. The default export deliberately keeps
the honest, round-trip-validated `OutTrans=0` order everywhere.

## A real export defect found and fixed: missing QK-norm (`qwen3` build)

`PICO_NORMS.md` §3 proves pico applies **per-head QK-norm** to Q and K (RMSNorm over `head_dim=64`;
48 of its 96 normalizations), with gamma = 1. The original export used the **`llama`** architecture,
which has **no QK-norm at all** -- so the exported model never applied that normalization and fed
unnormalized Q/K into the attention scores. That is a genuine functional defect, independent of any
weight-ordering question.

`tools/pico_to_gguf_qwen3.py` re-exports under llama.cpp's **`qwen3`** architecture -- RMSNorm +
per-head QK-norm + GQA + SwiGLU + RoPE, i.e. pico's exact recipe -- adding
`blk.N.attn_q_norm.weight` / `blk.N.attn_k_norm.weight` as ones (unit *gain*, but the normalization
now actually happens):

```
afmplus-v11.0-pico-qwen3-F16.gguf   1.14 GB   266 tensors   20 metadata keys
```

Verified: loads clean, and tokenization is confirmed correct
(`The capital of France is` -> `<bos> The(673) capital(5283) of(533) France(7005) is(567)`),
ruling the tokenizer out as a fault source.

**Output is still incoherent.** So QK-norm was a real defect but not the cause. Both this and the
`llama` build are kept: the `qwen3` one is architecturally faithful, the `llama` one is the historical
artifact. Neither is a working language model, for the reason established in
`PICO_POSREAD_RESULT.md` §20 -- the `O`/`down` output-channel ordering, which that section shows is
statistically indistinguishable from random against the residual basis.
