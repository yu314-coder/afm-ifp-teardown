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
| **Attention KV geometry** | ✅ **NKV=4 settled** (ablation: rank 53k vs 72k at NKV=8; live qkv is 3072-wide, Q2048+K512+V512) |
| **Norms** | ✅ **RMSNorm γ compile-time *folded* into linears (γ=1 correct); QK-norm[128] the only explicit γ (35 recovered)** |
| **Tokenizer** | ✅ working (byte-BPE, validated round-trip) |
| **Router** (`ExportableExpertSelector`, plain fp16) | ✅ extracted — but ⚠️ **proven UNNECESSARY** (false blocker) |
| **Expert-selection map** (which-of-219, "missing constant table") | ✅ **proven irrelevant AND ungated full-219 sum confirmed *optimal*** (selection/gating/scaling all score strictly worse) |
| **`metadata.bin` swizzler** | ✅ **decoded = two physical FFN address tables** (down-proj `0x0600beef`, gate/up `0x0a00b0ef`) |
| **down-proj experts** | ✅ **codec cracked = same as gate/up** (the missing factor was the per-block scale, not a z-order) |
| **Embedding + tied unembed** | ✅ exact (every token self-ranks 0) |
| **Attention forward** (all 44 layers) | ✅ runs stably (bounded residual growth) |
| **Coherent text generation** | 🔴 **blocked by an information limit** — assembled 44-layer forward carries real signal (3.6× chance) but the summed-FFN direction is mis-aligned, and per-layer activations are ANE-internal so it can't be validated/bisected — see below |
| **Running the real model** | ✅ via `afm` (Apple's `FoundationModels` runtime) |

---

## The current boundary — component-complete, blocked by absent ground truth

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

**A full 44-layer forward ablation (against Apple's emitted token) confirmed two of these and
re-located the remaining gap honestly.** Using a rank oracle (`"The capital of France is"` →
`▁Paris`; chance ≈ 76k of 152k):

- **NKV=4 is settled** (attention-only rank 53k vs 72k at NKV=8) and **the ungated full-219 sum is
  not just sufficient but *optimal*** — vs the full sum (rank 24.9k), skipping the sparse FFN gives
  122k, scaling it ×0.1 gives 36k, and top-14 activation-gating gives 39k. Every form of selection
  or attenuation is *worse*, corroborating the false-blocker result from the forward side.
- **But the assembled forward does not produce coherent text.** With the best conventions
  (interleaved RoPE, down-proj on the `[EH,D]` neuron axis, interleaved expert-region offsets,
  block gate/up fusion) it reaches rank ≈ 21k — **3.6× better than chance, still garbage.** The
  summed-FFN branch has norm ≈ 4.9×10⁴ against a residual norm ≈ 1.5×10², and dominates the output
  with a **mis-aligned (junk) direction**; every sandwich/post-norm placement is *worse* than a
  plain residual add, so it is a direction error (per-neuron gate/up/down alignment at the
  56,064-wide intermediate and/or per-layer expert offsets), not a magnitude one.

**The honest boundary is an *information* limit, not a mechanical one.** Fixing the FFN alignment
requires per-layer ground truth to bisect against — and Apple's runtime keeps every intermediate
activation ANE-internal (a search of an 8 GB IOSurface-targeted core for even the *input
embeddings* scores 0.17 = noise). So the alignment cannot be validated or debugged
component-by-component from any accessible artifact. This is the same class of wall reached
independently from the ANE/`.hwx` side; it needs either ANE-internal activation capture or a clean
single-geometry re-extraction, not further convention search.

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
