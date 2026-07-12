#!/usr/bin/env python3
"""extract_router.py — recover Apple's IFP expert-selection router from project_experts.mlasset.

The router (class `ExportableExpertSelector`) ships as its own asset and is NOT ANE-baked:
its weights are plain fp16 in one contiguous block, no palettization codec. This overturns the
earlier conclusion that the selector was compiled into the Neural Engine and unrecoverable.

Op graph (from the odix): in_expert_hidden_state
    -> RMSNorm(gamma) -> hidden_transform(1536->548) -> phi -> output_transform(->7008) -> topk
    reshape 7008 -> (32 layers, 219 experts); 10 active + 4 shared.
phi is identity/GELU (sigmoid is ruled out: it collapses the masks to be input-independent).

This script operates ONLY on the asset already present on an Apple device you own. It does not
contain or redistribute Apple's weights.

Usage:
    python3 extract_router.py /path/to/project_experts.mlasset/main-unspecialized.odix
"""
import sys, struct, numpy as np

D = 1536            # d_model
H = 548             # router hidden dim
NL = 32             # sparse layers
NE = 219            # experts per layer  (7008 = 32 * 219)
DATA = 0x12600      # start of the contiguous fp16 weight block (from region map)


def load(path):
    d = open(path, "rb").read()

    def fp16(off_elem, n):
        b = DATA + off_elem * 2
        return np.frombuffer(d[b:b + n * 2], dtype=np.float16).astype(np.float32)

    norm = fp16(0, D)                                   # folded gamma (~0-centred delta)
    hid = fp16(D, H * D).reshape(H, D)                  # hidden_transform  [548,1536]
    out = fp16(D + H * D, NL * NE * H).reshape(NL * NE, H)   # output [7008,548]
    tail_off = D + H * D + NL * NE * H
    tail = fp16(tail_off, (len(d) - DATA) // 2 - tail_off)   # biases (7084)
    return norm, hid, out, tail


def rms(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def selector(h, norm, hid, out, obias, phi="id"):
    """h: [1536] hidden state -> logits [32, 219]."""
    x = rms(h) * (1.0 + norm)            # RMSNorm with folded gamma
    z = hid @ x                          # [548]
    if phi == "gelu":
        z = 0.5 * z * (1 + np.tanh(0.79788456 * (z + 0.044715 * z ** 3)))
    logits = out @ z + obias             # [7008]
    return logits.reshape(NL, NE)


def mask_for_layer(logits_layer, k=10, shared=(0, 1, 2, 3)):
    top = set(np.argsort(-logits_layer)[:k].tolist())
    top.update(shared)
    return sorted(top)


if __name__ == "__main__":
    norm, hid, out, tail = load(sys.argv[1])
    obias = tail[:NL * NE]
    print("norm  [1536]      std=%.4f" % norm.std())
    print("hidden[548,1536]  std=%.4f rownorm=%.3f" % (hid.std(), np.linalg.norm(hid, 1).mean()))
    print("output[7008,548]  std=%.4f  reshape->(32,219)" % out.std())
    # sanity: masks must discriminate inputs (that is what makes it a real router)
    rng = np.random.RandomState(0)
    h1, h2 = rng.randn(D) * 0.5, rng.randn(D) * 0.5
    L1, L2 = selector(h1, norm, hid, out, obias), selector(h2, norm, hid, out, obias)
    m1, m2 = set(mask_for_layer(L1[0])), set(mask_for_layer(L2[0]))
    print("layer-0 mask overlap between two distinct inputs: %d/%d "
          "(low = discriminating = correct)" % (len(m1 & m2), len(m1)))
