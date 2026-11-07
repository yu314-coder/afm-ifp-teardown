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
| Fidelity | **91.7%** teacher-forced next-token agreement with Apple's live model (40 prompts, 1127 answer tokens) |
| Speed (M4) | ~150 tok/s prompt, 12–21 tok/s generation |

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
instructions), on 40 prompts:

| Expert routing | next-token agreement (teacher-forced) | first answer token | answers identical to Apple's |
|---|---|---|---|
| Apple's schedule (default) | **91.7%** | 35/40 | 7/40 |
| prompt per token, then Apple's schedule (`AFM_ROUTE_PROMPT_OWN=1`) | 91.3% | 38/40 | 9/40 |
| per token (`AFM_ROUTE_TOKEN=1`) | 90.5% | 38/40 | 11/40 |
| earlier one-turn template, per token | 87.5% | 30/40 | – |

"Answers identical" compares our free-running greedy answer with Apple's character for character; a single different
early token sends the rest of an answer elsewhere. Apple's schedule was confirmed independently: replaying 47 of Apple's
own expert re-picks, our router changes as many experts per re-pick as Apple's (27 vs 27 on average, r = 0.84).
On the 93 teacher-forced misses, Apple's token is our second choice 76% of the time and in our top five 98%; what
separates the port from Apple now is arithmetic (Apple's Neural Engine runs in fp16), which also nudges a few expert
picks — see the paper's "Toward 100%" section.

**What still differs, and why 100% is out of reach without Apple's hardware.** The weights, tokenizer, prompt format,
attention scale and expert schedule all match Apple's (each was checked separately; see the paper). What remains is
arithmetic: Apple runs the model on the Neural Engine in fp16 with its own lookup-table activations and summation
order. With 2-bit weights and an int8 KV cache, the model is sensitive to rounding: two mathematically identical fp32
implementations already disagree on 1–2.5% of next-token choices. Emulating Apple's int8 KV cache (+1.0 point) and its
32-token expert re-pick (+0.8) are included; rounding activations or RoPE angles to fp16 changes nothing measurable.

## What is in this repository

| Path | |
|---|---|
| `afm2gguf/afm2gguf.py` | standalone converter (NumPy + gguf-py), reads only the local assets |
| `llama.cpp-patch/` | the `afm` architecture for llama.cpp (graph, KV sharing, Apple's expert schedule, int8 KV emulation, Metal kernels) |
| `.github/workflows/build.yml` | builds Windows / Linux / macOS binaries from upstream llama.cpp + the patch |
| `tools/template_probe.swift` | checks the prompt template against Apple's `tokenCount` |
| `tools/afm.swift` | CLI / OpenAI-style server over Apple's live model (the fidelity reference) |
| `paper/` | the report (LaTeX source, figures, PDF) |
| `disclosure/` | correspondence with Apple about this work |

Earlier research notes and scripts (the v11.0/v11.1 work, the 300M draft model, the nano model) remain in the git
history; commit `e9a59b2` is the last one that contains them.

## Responsible use

The model weights belong to Apple. Do not redistribute the GGUF you produce. This project is unofficial and is not
affiliated with or endorsed by Apple; Apple has been asked whether redistribution would be permitted
(see `disclosure/`).
