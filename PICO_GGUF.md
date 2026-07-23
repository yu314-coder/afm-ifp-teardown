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
