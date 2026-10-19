# Reproducing the reconstruction on your own device

**Research / interoperability use only.** Everything here operates on the Foundation Models asset
that **already exists on your own Apple device**. Nothing in this repo is, or redistributes,
Apple's weights or tokenizer — those are Apple's proprietary property and stay on your machine.
This directory publishes the *method* (original research) so a researcher with their own hardware
can regenerate the decode locally. Use consistent with your local law and Apple's software license.

> **What you get is a negative result, by design.** The assembled model loads and runs in
> `llama.cpp` but does **not** predict text (it degenerates to copying) — see
> `paper/afm_teardown.pdf` → *The Reconstruction Gap: A Controlled Negative Result*. This pipeline
> reproduces that finding; it is a research instrument, not a working language model.

## Prerequisites
- An Apple Silicon Mac with Apple Intelligence enabled and the `GenerativeModels` asset downloaded.
- Python 3 with `numpy`.
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) built locally (for `gguf-py` and a runner).

## Step 0 — locate the asset on your device
```bash
zsh reproduce/find_asset.sh
```
This prints the paths to your local `ifp_rasterized_weights.bin` and `binary_0.hwx`. (The exact
`<hash>.asset` directory name differs per machine and per model version — the script finds it.)

## Step 1 — decode the sparse-expert pool (runnable here)
```bash
python3 reproduce/decode_experts.py \
  "/System/Library/AssetsV2/.../model.odixpackage/ifp/ifp_rasterized_weights.bin"
```
Self-contained, byte-exact (2-bit palette `[-1.5,-0.5,0.5,1.5]`, bank-cyclic out-axis
`m = 256*bank + 16*unit + local`). Verified to transfer across versions (v11.1 → v11.2). Import
`expert(layer, exp, which)` to get any gate/up/down tensor as float32.

## Step 2 — decode the backbone, norms, and embedding
The attention/dense weights (from `binary_0.hwx`), the per-layer normalisation stream (from the
program's `__const`), and the int4 embedding are specified **byte-exactly in the paper**:
- `paper/` → *Weight recovery* and *Model specification* — the ANE de-swizzle, the striped out-axis,
  the per-output-row scales, and the gamma/QK-norm stream.
- the tracked decoders in `src/` and `tools/` (e.g. `src/deswizzle.py`, `src/ane_embed_codec.py`,
  `src/odix.py`) implement the codec primitives these use.

> Note: the end-to-end backbone decoder was validated on **v11.1**. On **v11.2** the expert pool
> transfers unchanged (Step 1), but the backbone's on-disk framing changed (see the paper's
> execution-boundary section); re-deriving it is the open item. The paper is the authoritative spec.

## Step 3 — assemble the GGUF
`build_gguf.py` (see the paper for the exact tensor map) writes a `gemma3`-architecture GGUF with
layers 0–34 (12 dense + 23 sparse), uniform `n_ff=11776`, the recovered tokenizer (`+4` offset),
and the embedding fidelity-scaled for gemma3's `sqrt(d)` convention. Output stays local.

## Step 4 — run it
```bash
# with llama.cpp built with Metal (arm64):
./build/bin/llama-completion -m /path/to/afm-ifp3-L35-f16.gguf \
    -p "The capital of France is" -n 24 --temp 0

# or the pure-numpy forward (third independent engine; same result):
GGUF_PY=/path/to/llama.cpp/gguf-py python3 reproduce/numpy_forward.py /path/to/afm-ifp3-L35-f16.gguf
```
Both produce the documented copy-machine output. That reproduction — three independent engines, same
failure from the same recovered weights — is the point: it localises the defect to a residual-axis
weight-value correspondence that no shipped byte reveals, exactly as the paper argues.
