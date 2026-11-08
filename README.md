# afm-ifp-teardown — Apple's on-device foundation model in llama.cpp

A from-weights reconstruction of the language model behind Apple Intelligence on macOS 27
(`SystemLanguageModel`, display version **`afmplus-v11.2-ifp`**, the sparse on-device model, "AFM 3 Core Advanced"),
packaged as a GGUF that runs in llama.cpp on macOS, Windows and Linux.

This repository contains **code only**. It does not contain, and will never contain, Apple's weights, tokenizer, decoded
assets or a GGUF. The converter rebuilds the GGUF from the model files Apple Intelligence has already installed on
your own Mac.

| | |
|---|---|
| Model | 56 layers (12 dense + 44 IFP-sparse), d=1536, 16 Q / 8 KV heads × 128, KV sharing in the last 21 layers, 20.3 B parameters stored |
| Weights | Apple's 2-bit codes and fp16 scales carried into Q2_K **bit-exactly**; int4 embedding as Q4_0 (exact) |
| Prompt format | Apple's `afm_cider_v2` template, decoded from the runtime and verified against `SystemLanguageModel.tokenCount` |
| Expert routing | Apple's schedule: experts picked from the last prompt token, re-picked every 32 positions |
| Fidelity | **97.0%** teacher-forced next-token agreement with Apple's live model (40 prompts, 1127 answer tokens; 95.3% on a harder 230-prompt set); 17/40 answers identical character for character |
| Speed (M4, Metal) | ~150 tok/s prompt, ~28 tok/s generation (`AFM_EXACT_SILU=1`: ~290 / ~35 tok/s, slightly less faithful) |

The full write-up (architecture diagrams, storage formats, the bug that hid the model, validation, every negative
result) is in [`paper/afm_teardown.pdf`](paper/afm_teardown.pdf).

## 1. Get a llama.cpp build with the `afm` architecture

**Windows / Linux / macOS binaries:** download a zip from this repository's
[Actions / Releases](../../releases) (`win-x64-cpu`, `win-x64-vulkan`, `win-x64-cuda12`, `linux-x64-cpu`,
`linux-x64-vulkan`, `macos-arm64-metal`). They are plain llama.cpp builds with the patch below applied.

**Or build it yourself** (any OS):
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 41ea26144e55d23f37bb765f88c07588d786567f       # the patch's base
git apply /path/to/afm-ifp-teardown/llama.cpp-patch/llama.cpp-afm.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF    # add -DGGML_CUDA=ON or -DGGML_VULKAN=ON as needed
cmake --build build -j
```

## 2. Convert (on a Mac with Apple Intelligence; ~3 min, 6.3 GiB)
```bash
pip install numpy gguf
python3 afm2gguf/afm2gguf.py -o afm-v11.2-ifp.gguf
```
The converter finds the model through the AssetsV2 catalog and checks everything it decodes (container layout,
palette, router mask, tokenizer, norm-gain alignment) before writing. If the model files are not mounted it prints
the `hdiutil attach` command for the asset's cryptex image. The resulting file is portable: copy it to any machine.

## 3. Run
```bash
llama-cli    -m afm-v11.2-ifp.gguf --jinja -st --temp 0 -p "Why is the sky blue?"
llama-server -m afm-v11.2-ifp.gguf --jinja          # OpenAI-compatible API; a system message = Apple's "instructions"
```

## How faithful is it?

Measured against Apple's own model through the public FoundationModels API (`tools/afm.swift`, temperature 0, the same
instructions), on 40 prompts (reproduce with [`tools/eval`](tools/eval)):

| Version / expert routing | next-token agreement (teacher-forced) | first answer token | answers identical to Apple's |
|---|---|---|---|
| **v1.3**, as v1.2 + no input bias at re-pick tokens (default) | **97.0%** | 38/40 | **17/40** |
| v1.2, Apple's schedule + Neural Engine SiLU | 96.7% | 38/40 | 17/40 |
| v1.1 = v1.2 with `AFM_EXACT_SILU=1` | 96.1% | 37/40 | 17/40 |
| v1.1, per token (`AFM_ROUTE_TOKEN=1`) | 93.8% | 36/40 | 12/40 |
| v1.0, Apple's schedule | 91.7% | 35/40 | 7/40 |
| v1.0, per token | 90.5% | 38/40 | 11/40 |

**What changed in v1.1:** two layer-0 details.
- **Embedding scale.** llama.cpp's generic embedding code already applied the GGUF's `embedding_scale`, and the AFM graph applied it a second time, so the model ran with its input embedding ×100. Apple applies no multiplier at all.
- **Input bias.** Apple's prompt procedures add nothing at layer 0, while its extend procedures (generated tokens) add the input bias; the port now does the same.

**What changed in v1.2:** the FFN activation uses Apple's Neural Engine SiLU by default. It is a 33-knot piecewise-linear table (fp16 output); Apple's program contains the identical table once per FFN task. We measured it on the Neural Engine through Core ML and reproduce it to within one fp16 step for every input. `AFM_EXACT_SILU=1` switches back to the exact function.

**What changed in v1.3:** at a re-pick (every 32nd position), Apple runs a dense-only function on that token, and like the prompt procedures it adds no layer-0 bias. The port now does the same. On the 230-prompt set this is worth +0.08 points (12 tokens fixed, 7 broken), and it brings our expert-change counts slightly closer to Apple's.

**Router precision:** the router decides experts by margins around 0.001, smaller than the error of some GPU matrix kernels (Metal's deviate by about 0.0001). Its small weights now stay on the CPU, so expert picks are exact and identical on every backend. That costs a few percent of speed, and `AFM_ROUTER_GPU=1` restores the GPU router. Agreement is unchanged within noise (6256 vs 6261 of 6572 tokens).

Old GGUFs work with the current patch; nothing needs to be re-converted.

"Answers identical" compares our free-running greedy answer with Apple's character for character; a single different
early token sends the rest of an answer elsewhere. Apple's schedule was confirmed independently: replaying 47 of Apple's
own expert re-picks, our router changes as many experts per re-pick as Apple's (27 vs 27 on average, r = 0.84).

**A larger test (230 prompts, 6572 answer tokens).** The 40-prompt set moves by about ±0.4% from noise, so every candidate was re-run on 190 more prompts (`tools/eval/prompts_extended.txt`):

| Variant (teacher-forced, 6572 tokens) | agreement | vs default: fixed / broken |
|---|---|---|
| **v1.3 default** | **95.27%** | 12 / 7 vs v1.2 |
| v1.2 default | 95.19% | – |
| exact SiLU (`AFM_EXACT_SILU=1`) | 94.17% | 72 / 139 |
| per-token routing | 93.34% | – |
| fp16 rounding of every matmul output (`AFM_MM_F16=1`) | 95.13% | 30 / 34 |
| int8 KV emulation off | 95.07% | 46 / 54 |
| int8 rounding by truncation instead of to-nearest | 94.05% | 64 / 139 |
| keys quantised before their gain | 95.09% | 43 / 50 |
| RoPE tables rounded to fp16 | 94.99% | 21 / 34 |
| re-picks that favour already-loaded experts | 95.15% | – |

Apple's log also confirms the re-pick timing: around a boundary its runtime steps one token at a time, then logs "Picking IFP experts at position=96".

**Why 100% is out of reach from outside.** We rebuilt the 12 dense layers as a Core ML program on the real Neural Engine and injected its layer-12 state into llama.cpp. Agreement stayed the same (94.9% vs 95.2%), but 3.4% of tokens flipped in both directions. Two plausible versions of the hardware arithmetic disagree on as many tokens as separate us from Apple, and Core ML does not compile our graph into Apple's hand-built Neural Engine program. The last few percent are therefore set by Apple's exact rounding, not by anything missing from the model.

**Apple's expert decisions, measured.** Apple logs each pick, but the expert list is redacted as sensitive. Its public cache counter (`replaced=` at each re-pick) still shows how many experts changed. Our router predicts those counts with r = 0.93 and a mean miss of 4.6 out of about 40 changed experts. The miss is the same, within noise, for every arithmetic variant, including the state computed on the Neural Engine. So our expert picks are within a few of Apple's 1012 per pick, and no rounding model we can build gets closer.

**What still differs.** 37 of 1127 answer tokens on the 40-prompt set (316 of 6572 on the larger one; 2.9% of tokens are missed whatever the routing, 1.9% only with Apple's schedule). Apple's token is almost always our second or third choice. The weights, tokenizer, prompt format, layer-0 input, attention scale and expert schedule all match Apple's; each was checked separately (see the paper).

Measured directly and ruled out as the main cause:
- **The Neural Engine's arithmetic.** Its matrix multiplies are accurate (high-precision accumulation, fp16 output). Its SiLU table is now emulated, which is worth +0.6 points. Rounding the activations to fp16 changes nothing measurable.
- **Apple's speculative decoding.** The tolerance is 0, so it is exact.
- **The FM-API adapter.** It is declared empty.
- **A repetition penalty.** There is none.

The remaining misses are near-ties: Apple's token is our second choice in 84% and in our top three in all; the median logit gap is 0.4. The same model on CPU and on Metal (identical mathematics) already disagrees on 0.7% of answer tokens. Apple's Neural Engine computes in fp16 with approximate softmax, exp and normalisation, so its rounding differs from ours by more than that. The last few percent would need bit-exact Neural Engine emulation; rounding every matmul output to fp16 (`AFM_MM_F16=1`) moves the score only within noise (96.9%).

## What is in this repository

| Path | |
|---|---|
| `afm2gguf/afm2gguf.py` | standalone converter (NumPy + gguf-py), reads only the local assets |
| `llama.cpp-patch/` | the complete llama.cpp side of the project as one patch against upstream `41ea26144`: the `afm` architecture (graph, KV sharing, Apple's expert schedule, int8 KV emulation, Metal kernels) and the `llama-afm-logits` evaluation tool |
| `tools/eval/` | reproduces the fidelity numbers: collects Apple's answers, then teacher-forced and free-running comparisons |
| `.github/workflows/build.yml` | builds Windows / Linux / macOS binaries from upstream llama.cpp + the patch |
| `tools/template_probe.swift` | checks the prompt template against Apple's `tokenCount` |
| `tools/afm.swift` | CLI / OpenAI-style server over Apple's live model (the fidelity reference) |
| `paper/` | the report (LaTeX source, figures, PDF) |
| `disclosure/` | correspondence with Apple about this work |

Research notes and scripts from before v1.0 (the v11.0/v11.1 work, the 300M draft model, the nano model) remain in the git
history; commit `e9a59b2` is the last one that contains them.

## Responsible use

The model weights belong to Apple. Do not redistribute the GGUF you produce. This project is unofficial and is not
affiliated with or endorsed by Apple; Apple has been asked whether redistribution would be permitted
(see `disclosure/`).
