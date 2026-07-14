# Apple On-Device Foundation Model — IFP Teardown & Reconstruction

A complete static reverse-engineering of Apple's on-device **`afmplus-v11.0-ifp`** model — the
sparse Mixture-of-Experts language backbone shipped in
`com.apple.MobileAsset.UAF.FM.GenerativeModels` (macOS 26/27) — from container formats down to a
component-validated PyTorch forward pass, plus a working CLI/server over Apple's runtime.

> **This repo contains original research only: the teardown paper and the decode/inference
> code. It does NOT contain, and will not distribute, Apple's weights or tokenizer data** — that
> is Apple's copyrighted, proprietary property. The code operates on the asset files that
> already exist locally on an Apple device. Use for interoperability/research on hardware you
> own, consistent with your local law and Apple's license.

---

## TL;DR — what was found

The model is a **sparse MoE**: ~9.86 B stored params (12 dense + 32 sparse layers, d=1536, GQA,
RoPE θ=500000, SwiGLU, 219 routable experts/layer with Instruction-Following Pruning; the shipped
default `ifp1_r48` keeps 10 active + 4 shared). Everything Apple ships as **data** is recovered
and validated.

The headline result of the latest pass is a **correction**: the expert-selection
router→physical-expert map — long believed the fatal, unrecoverable blocker — is **mathematically
irrelevant**. The IFP sparse FFN is an *ungated* structured-pruned SwiGLU (a permutation-invariant
sum over experts), so no selection map is needed to run it.

| Component | Status |
|---|---|
| **All 9.86 B weights** | ✅ recovered, 100% of file, verified |
| **Container stack** (Cryptex, BBBB, odix, hwx, MPSGraph) | ✅ fully characterized |
| **Weight codec** (4-bit LUT + per-1024 fp16 block scale + ANE 8×128 de-swizzle) | ✅ validated |
| **Dense linear weights** | ✅ **validated against coreml2hwx ground truth** — real-weight decay 8–11 (scrambled ≈1.45) |
| **Architecture** (32 full-qkv + 12 kv-reuse attn; 12 dense + 32 expert FFN) | ✅ recovered |
| **Norms** | ✅ **RMSNorm γ compile-time *folded* into linears (γ=1 correct); QK-norm[128] the only explicit γ (35 recovered)** |
| **Tokenizer** | ✅ working (byte-BPE, validated round-trip) |
| **Router** (`ExportableExpertSelector`, plain fp16) | ✅ extracted — but ⚠️ **proven UNNECESSARY** (false blocker) |
| **Expert-selection map** (which-of-219, "missing constant table") | ✅ **proven irrelevant** (ungated permutation-invariant sum) |
| **`metadata.bin` swizzler** | ✅ **decoded = two physical FFN address tables** (down-proj `0x0600beef`, gate/up `0x0a00b0ef`) |
| **down-proj experts** | ✅ **codec cracked = same as gate/up** (the missing factor was the per-block scale, not a z-order) |
| **Embedding + tied unembed** | ✅ exact (every token self-ranks 0) |
| **Attention forward** (all 44 layers) | ✅ runs stably (bounded residual growth) |
| **Coherent text generation** | 🟡 pending the mechanical per-layer sparse-FFN assembly — see below |
| **Running the real model** | ✅ via `afm` (Apple's `FoundationModels` runtime) |

---

## The current boundary — component-complete, one mechanical step from text

The model is **component-complete**: weights, codec, de-swizzle, architecture, norms, tokenizer,
and — critically — the fact that *no expert-selection map is needed* are all established. Several
earlier "walls" collapsed on inspection:

- **The router→expert map is a false blocker.** The runtime graph's `ANE_IFPLayerSequence` op has
  no router, top-k, gather, scatter, or softmax-over-experts; weights enter through a fixed symbol.
  A non-gated SwiGLU is a permutation-invariant sum `FFN(h) = Σᵢ SiLU(gᵢ·hₙ)(uᵢ·hₙ)·dᵢ`, so the
  logical which-of-219 order does not affect the output. Summing the resident set is the correct
  pruned FFN; summing the full superset recovers the un-pruned base model. (The earlier
  "~17× overcount" was an artifact of a mis-specified *gated* interpretation.)
- **The down-proj was never a different format.** It uses the same 4-bit → codebook → per-1024
  fp16 block-scale codec as gate/up. The prior "down-proj = noise" verdict was a mis-calibrated
  low-rank metric; the omitted per-block scale was the missing factor.
- **The norms aren't missing — they're folded.** The shipped graph has 661 *parameter-free*
  RMSNorm ops; each learned γ is folded into the adjacent linear at ANE-compile time, so γ=1 is
  the correct runtime. Only the per-head QK-norm γ∈ℝ¹²⁸ survives explicitly (35 recovered from a
  live heap).
- **The dense weights are validated without Apple's activations.** Compiling a positional-probe
  weight through Apple's own `coreml2hwx` toolchain and reading back the byte permutation confirms
  the de-swizzle; the recovered dense weights carry genuine low-rank structure (decay 8–11).
- **Intermediate activations are ANE-internal.** A search of an 8 GB IOSurface-targeted process
  core for the input embeddings scores 0.17 (noise): nothing crosses to host DRAM, so the forward
  is validated end-to-end (against Apple's emitted token), not per-layer.

**What remains is mechanical, not an information wall:** the *per-layer physical assembly* of the
resident-expert FFN — gather each sparse layer's gate/up/down blocks from the expert region with
the block scale and sum them ungated, then unembed the final (post-layer-44) hidden. The prior
version of this project reported the down-proj micro-tile z-order as the sole blocker to coherent
text; that has been superseded — the wall was largely a *validation-target error* (unembedding an
intermediate hidden state) plus the false-blocker router map, not missing data.

Full consolidated record: see the teardown paper §"From-Weights Reconstruction".

---

## What's here

| Path | Contents |
|---|---|
| `paper/afm_teardown.pdf` | Full teardown paper (AMS-style, 15 pp.) — compile from `.tex` |
| `FINDINGS.md` | Condensed technical findings (layout, codec, MLIR/odix parse) |
| `ROUTER_EXTRACTION.md` | The `ExportableExpertSelector` extraction (now proven unnecessary to run) |
| `ODIX_DECOMPILER.md` | `main-h16g.odix` structural map (38 configs, op format) |
| `src/afm.swift` | **`afm`** — CLI + OpenAI-compatible server over Apple's real model |
| `src/afm_tokenizer.py` | Byte-BPE tokenizer (validated) |
| `src/deswizzle.py`, `crack_lut.py` | ANE de-swizzle + LUT codec + structure-ratio metric |
| `src/odix.py` | Parser for Apple's `odix` container |
| `src/rebuild_full_pt.py` | Decode → single-file weight export |
| `src/afm_forward_working.py` | Reconstructed forward pass (attn + ungated MoE-SwiGLU) |
| `src/afm_generate.py` | End-to-end generation harness (over Apple's runtime) |

---

## Using the real model — `afm` (recommended)

This runs Apple's actual model, correct ANE routing and all — the faithful way to *use* it.

```bash
swiftc -O -o afm src/afm.swift          # macOS 26+ with Apple Intelligence

./afm "What is the capital of France?"   # → The capital of France is Paris.
echo "prompt" | ./afm                     # piped
./afm -s "You are a pirate." -t 0.9 "Hi" # system prompt + temperature
./afm --stream "Write a haiku"            # token streaming
./afm                                     # interactive REPL
./afm --server 8080                       # OpenAI-compatible API on :8080
#   -> POST /v1/chat/completions  (any OpenAI client works)
```

## Reproducing the teardown (on your own device)

1. From macOS **Recovery** Terminal (the live system seals the asset cache), copy the model
   assets to external storage — see the paper §"Recovery Pipeline" for the `cp -R` of
   `.../AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels`.
2. `hdiutil attach -readonly` the IFP Cryptex; point `rebuild_full_pt.py` at
   `model.odixpackage/ifp/ifp_rasterized_weights.bin`.
3. It decodes, R-verifies each tensor, and writes a single `state_dict` (audit: 100.00% of file,
   0 non-finite). The forward harness runs the reconstructed model.

---

## License

Original research and code: MIT. **Apple's model weights and tokenizer data are not included and
not covered by this license.** Interoperability/research use on your own hardware only.
