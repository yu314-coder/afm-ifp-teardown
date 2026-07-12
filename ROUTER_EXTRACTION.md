# The IFP router — extracted, decoded, validated

This is the headline result of the teardown, and it **corrects an earlier conclusion**.

## Earlier claim (wrong) → corrected

An earlier pass concluded the expert-selection router was *baked into the Neural Engine at
export time* — that instruction-following pruning collapsed the per-instruction mask into a
fixed dense set and **no runtime mask predictor shipped**. That was wrong. **The router ships as
its own asset and is fully extractable as plain fp16.**

## Where it lives

`model.odixpackage/ifp/project_experts.mlasset/main-unspecialized.odix`

Its `odix` op graph names the class **`ExportableExpertSelector`**, with I/O
`in_expert_hidden_state → out_expert_logits` and the op sequence:

```
RMSNorm(γ) → hidden_transform → φ → output_transform → sigmoid → topk
```

There is **no dequantize/gather op** — the weights are stored as **plain fp16**, one contiguous
block from `0x12600 → EOF`. No palettization codec is involved (unlike the FFN experts).

## The three tensors (grounded offsets, fp16)

| tensor | element offset from `0x12600` | shape | role |
|---|---|---|---|
| `norm_weight` (folded γ) | 0 | `[1536]` | RMSNorm gain (stored as ~0-centred δ; use γ = 1+δ) |
| `hidden_transform` | 1536 | `[548, 1536]` | project d_model → H=548 |
| `output_transform` | 1536 + 841728 | `[7008, 548]` → `[32, 219]` | logits: 32 layers × 219 experts |
| biases (tail) | end | `7084` | output bias = first 7008 |

`7008 = 32 layers × 219 experts` reshapes with **uniform per-layer norms (3.5–3.8)** — the layout
self-verifies. `H=548` is forced by the total element count.

## The selector

```
mask_layer = top_10( output_transform · φ( hidden_transform · RMSNorm_γ(h) ) )  + 4 shared
```

recomputed every 32 tokens (`expert_selection_frequency`), per the config's `active_experts:10`,
`shared_experts:4`.

## Validation — it behaves like a router

Feeding controlled hidden states (`extract_router.py`):

| activation φ | mask overlap between two distinct inputs | verdict |
|---|---|---|
| **identity / GELU** | **1–2 / 10** (171 distinct experts across 32 layers) | discriminating ✓ |
| sigmoid | 9 / 10 (input-independent) | ruled out as hidden φ |

Deterministic per input, discriminating across inputs, healthy expert spread. This is Apple's
router, **extracted — not a trained mimic.**

## Reproduce

```bash
python3 src/extract_router.py \
    /Volumes/.../project_experts.mlasset/main-unspecialized.odix
```

(Operates on the asset already on your own device; ships no Apple weights.)
