# Apple On-Device Foundation Model — IFP Teardown

A static reverse-engineering teardown of Apple's on-device generative model
**`afmplus-v11.0-ifp`** (the sparse Mixture-of-Experts language backbone shipped in
`com.apple.MobileAsset.UAF.FM.GenerativeModels` on macOS 26/27), documenting its
architecture, weight codec, and container formats — and a validated pipeline that decodes
the shipped weight file into a single named `state_dict`.

> **This repository contains only original research: the teardown paper and the decode
> code. It does _not_ contain, and will not distribute, Apple's model weights.** The weights
> are Apple's copyrighted, proprietary property. The code here operates on the asset files
> that already exist locally on an Apple device; it does not download or redistribute them.
> Use for interoperability/research on hardware you own, consistent with your local law and
> Apple's license.

## What's here

| Path | Contents |
|---|---|
| `paper/afm_teardown.pdf` | The full teardown paper (AMS-style, 12 pp.) — compile from `.tex` |
| `FINDINGS.md` | Condensed technical findings (layout, codec, MLIR parse) |
| `src/deswizzle.py` | The ANE 8×128 tile de-swizzle (the key to weight recovery) |
| `src/crack_lut.py` | LUT-palettization codec + structure-ratio validation metric |
| `src/odix.py` | Parser for Apple's `odix` on-device executable container |
| `src/rebuild_full_pt.py` | End-to-end decoder → single-file model export |

## The findings, in brief

- **Architecture** (from `config.json`): 44 layers (12 dense + 32 sparse), d=1536, GQA
  (Q=2048, KV=1024, head 128), RoPE θ=500000, SwiGLU, RMSNorm; MoE with ~219 experts/layer,
  10 active + 4 shared. **Total ~9.86 B stored params; ~1.4 B active per instruction** — this
  is why it's an MoE built from a "3B"-class base.
- **Codec**: 4-bit index → 16-entry `lut_to_dense` codebook × per-1024-block fp16 scale,
  **then an ANE 8×128 tile de-swizzle** (`W = v.reshape(Co/8,Ci/128,128,8).transpose(0,3,1,2).reshape(Co,Ci)`).
- **Validation**: a scale-decontaminated low-rank structure ratio `R = σ₁(W)²/σ₁(shuffle(W))²`
  (genuine weights ≫ 1, noise ≈ 1). Palettization alone gives R≈3; **adding the de-swizzle
  gives R = 13–69** — genuine weight structure.
- **Resolved unknowns**: RMSNorm γ is **folded** into adjacent weights; the router is **baked**
  (the exported adapter is `prompt_opt_dense_only`); `binary_0.hwx` is **ANE microcode, not
  weights** (R≈1); the missing `ifp_constant_table_*.json` has a shipped runtime equivalent in
  `metadata.bin`.
- **Container stack**: Cryptex DMG → `BBBB` rasterized-weight file + `metadata.bin` swizzler,
  `odix` executable (self-relative refs), `MLIR22.0.0` MPSGraph bytecode, `0xBEEFFACE` ANE hwx.

## Reproducing (on your own device)

1. Obtain the assets from the macOS **Recovery** Terminal (the live system doesn't expose the
   cache): the author uses a small copy script run as `/Volumes/D/copyfm` that copies the
   `MobileAsset.UAF.FM.GenerativeModels` tree — including the Cryptex disk images — to
   external storage.
2. `hdiutil attach -readonly` the IFP Cryptex; point `rebuild_full_pt.py` at
   `model.odixpackage/ifp/ifp_rasterized_weights.bin`.
3. The script decodes, validates each tensor's R, and writes a single `state_dict`. It prints a
   coverage audit against the physical file (expected: 100.00%, 0 non-finite).

## License

Original research and code in this repo: MIT. **Apple's model weights are not included and are
not covered by this license.**
