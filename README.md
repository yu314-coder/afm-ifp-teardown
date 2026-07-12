# Apple On-Device Foundation Model — IFP Teardown & Reconstruction

A complete static reverse-engineering of Apple's on-device **`afmplus-v11.0-ifp`** model — the
sparse Mixture-of-Experts language backbone shipped in
`com.apple.MobileAsset.UAF.FM.GenerativeModels` (macOS 26/27) — from container formats down to a
running (if not-yet-calibrated) PyTorch forward pass, plus a working CLI/server over Apple's
runtime.

> **This repo contains original research only: the teardown paper and the decode/inference
> code. It does NOT contain, and will not distribute, Apple's weights or tokenizer data** — that
> is Apple's copyrighted, proprietary property. The code operates on the asset files that
> already exist locally on an Apple device. Use for interoperability/research on hardware you
> own, consistent with your local law and Apple's license.

---

## TL;DR — what was found

The model is a **sparse MoE**: ~9.86 B stored params, ~1.4 B active per instruction (12 dense +
32 sparse layers, d=1536, GQA, RoPE θ=500000, SwiGLU, 219 experts/layer with Instruction-
Following Pruning selecting 10 active + 4 shared). Everything Apple ships as **data** was
recovered, **including the expert-selection router** — see [`ROUTER_EXTRACTION.md`](ROUTER_EXTRACTION.md).

| Component | Status |
|---|---|
| **All 9.86 B weights** | ✅ recovered, 100% of file, verified |
| **Container stack** (Cryptex, BBBB, odix, hwx, MPSGraph) | ✅ fully characterized |
| **Weight codec** (4-bit LUT + per-block scale + ANE 8×128 de-swizzle) | ✅ validated (structure R, sane weights) |
| **Architecture, norm values, RoPE** | ✅ recovered |
| **Tokenizer** | ✅ working (byte-BPE, validated round-trip) |
| **Router** (`ExportableExpertSelector`, plain fp16) | ✅ **extracted + decoded + validated** — [`ROUTER_EXTRACTION.md`](ROUTER_EXTRACTION.md) |
| **`main-h16g.odix` decompiler** (38 configs, op format) | ✅ structural map — [`ODIX_DECOMPILER.md`](ODIX_DECOMPILER.md) |
| **FFN per-constant layout** (variable-width, pruned) | ⚠️ scoped: type/symbol schema-recovery remaining |
| **Full forward pass** (attn + MoE-SwiGLU + router + embed + head) | ✅ **runs stably end-to-end** |
| **Coherent text generation** | ❌ needs FFN layout + 3 calibration pieces — see "The honest limit" |
| **Running the real model** | ✅ via `afm` (Apple's `FoundationModels` runtime) |

---

## The honest limit — what remains before coherent text

The forward pass is numerically stable but does not yet emit Apple's text. Of the pieces once
thought blocking, **the router is now solved** (extracted, not a surrogate). The remaining gap is
concrete and scoped:

1. **FFN per-constant layout** — the 396 expert constants are shipped **IFP-pruned to variable
   widths** (42–232 experts/constant, *not* uniform). Their exact sizes require resolving the
   `main-h16g.odix` type/symbol tables (a bounded FlatBuffer schema-recovery problem — see
   [`ODIX_DECOMPILER.md`](ODIX_DECOMPILER.md)). A uniform-width model mis-aligns and is why the
   reconstructed FFN produces noise.
2. **Embedding dequant**, **output-norm γ**, **layer pairing** — three calibration pieces still
   approximate.

**Corrected verdicts along the way (the honest part):** several conclusions reversed under
scrutiny — the FFN wiring is a sequential layout (not ANE-baked); and, most importantly, the
**router is not ANE-locked at all** — it ships as plain fp16 in `project_experts.mlasset` and is
now extracted and validated. This project moved the model from "a black box baked into the ANE"
to "the router is in hand and the FFN layout is a precisely-scoped schema problem."

---

## What's here

| Path | Contents |
|---|---|
| `paper/afm_teardown.pdf` | Full teardown paper (AMS-style, 14 pp.) — compile from `.tex` |
| `FINDINGS.md` | Condensed technical findings (layout, codec, MLIR/odix parse) |
| `src/afm.swift` | **`afm`** — CLI + OpenAI-compatible server over Apple's real model |
| `src/afm_tokenizer.py` | Byte-BPE tokenizer (validated) |
| `src/deswizzle.py`, `crack_lut.py` | ANE de-swizzle + LUT codec + structure-ratio metric |
| `src/odix.py` | Parser for Apple's `odix` container |
| `src/rebuild_full_pt.py` | Decode → single-file weight export |
| `src/afm_forward_working.py` | Reconstructed forward pass (attn + MoE-SwiGLU, sequential FFN wiring) |
| `src/afm_generate.py` | End-to-end generation harness (runs; output not yet calibrated) |

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
   0 non-finite). `afm_forward_working.py` / `afm_generate.py` run the reconstructed model.

---

## License

Original research and code: MIT. **Apple's model weights and tokenizer data are not included and
not covered by this license.** Interoperability/research use on your own hardware only.
