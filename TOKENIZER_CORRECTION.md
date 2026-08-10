# Correction: the vocabulary was off by four

*2026-08-10.* A tokenizer scrape used throughout the earlier part of this project is **shifted by
four token ids**. This note records the error, the proof, what it invalidates, and what it does not.

## The error

Apple's tokenizer asset stores an offset array with **five leading entries** that are the model's
special-token *declarations*, not vocabulary:

```
[" ⁇ " (unk_surface), "<unk>", "<bos>", "<eos>", "<pad>"]
```

so the correct lookup is

```
piece(id) = string_at(BLOB + offset[id + 5])
```

`local/afm_odix/tok_vocab.json` was built with a shift of **+1**. Every entry in it therefore names
the token four ids higher than the one it labels:

| word | true id | id in the scrape |
|---|---|---|
| `▁Paris` | 9079 | 9083 |
| `▁dog` | 4799 | 4803 |
| `▁Monday` | 8492 | 8496 |

## Proof of the correct shift

Three independent structural checks, all re-derivable from the asset by
`tools/afm_tokenizer_asset.py --check`:

1. **The type array disagrees with +1.** `type[0..2] == 3` (CONTROL) and `type[3] == 2` (UNKNOWN).
   Only `+5` puts `<pad>`, `<eos>`, `<bos>` on the CONTROL ids and `<unk>` on the UNKNOWN id.
2. **The BYTE class only closes at +5.** The 256 type-6 entries occupy ids 238..493. At `+5` they
   read as a contiguous `<0x00>`..`<0xFF>`; at `+1` they read `</li>`..`<0xFB>`.
3. **The first NORMAL id reads `▁t`** — BPE merge rank 0 — at `+5`.

## What this invalidates

**The `▁Paris`-rank oracle.** Any test that looked a token up *by name* through the scrape and then
indexed logits by that number was reading a logit four positions away, belonging to an unrelated
token. This is the most likely explanation for the false positive already recorded against that
oracle, and it is a stronger reason to distrust it than the one previously given.

**The published pico GGUF vocabularies.** `local/pico_gguf/*.gguf` carry the shifted tokens; their
`tokenizer.ggml.scores` are all zero (real scores were never extracted) and their `token_type` has
only two distinct values, so the BYTE class — and with it byte fallback — is absent. Their
`eos_token_id = 110` is `106 + 4`, i.e. it was aiming at `<end_of_turn>` through the same shift.
`tools/pico_to_gguf.py` and `tools/pico_to_gguf_qwen3.py` have been corrected to read the asset
directly; the previously built files are stale and should be rebuilt.

A duplicate-token hack in the old writer — "ids 4-7 mirror `<pad>`/`<eos>`/`<bos>`/`<unk>`" — was a
*symptom* of the shift. At the correct offset there are no duplicate specials.

## What this does NOT invalidate

**The pico embedding decode is correct, and indexes by true id.** Nearest neighbours computed on the
decoded table, using true ids, put every probe word at rank 0 against itself; using the shifted ids
the word does not appear at all. So the shift is in the labels, not in the weights.

**The full-logit correlation oracle is unaffected.** It compares whole logit vectors between a
captured and a computed forward pass. A relabelling of the vocabulary axis is invisible to it,
because both sides use the same axis.

**No ordering or layout conclusion depends on this.** The container, palette, scale maps, block
layout, and residual orderings were all established from weight statistics, never from token names.

## A separate, unrelated caution

Beyond the shift, pico's decoded embedding gives a *self*-match at rank 0 but no semantically
related neighbours — its nearest neighbours after itself are noise. nano's embedding, decoded by the
same method, gives real ones (`▁Paris` → `Paris`, `▁París`, `▁France`, `▁London`, `▁Berlin`). So
nano's embedding is semantically validated and **pico's is not**; "self-ranks 0" alone is a vacuous
test, as already noted elsewhere in this repository.

## The asset format

Recovered by structural analysis; it is not a SentencePiece protobuf. The layout closes with zero
slack for both V = 262144 and V = 328192. See `tools/afm_tokenizer_asset.py` for the parser and the
full field table.
