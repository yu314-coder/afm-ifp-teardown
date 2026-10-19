#!/usr/bin/env python3
"""numpy_forward.py -- a pure-numpy forward pass over a reconstructed ifp3 GGUF.

RESEARCH / INTEROPERABILITY USE ONLY. Reads a GGUF you built locally from your own device's
assets (see build_gguf.py / REPRODUCE.md). Redistributes nothing. This reproduces, in a third
independent implementation, the negative result documented in paper/ (The Reconstruction Gap):
the assembled forward loads and runs but does not predict text. It is a research instrument, not
a working language model.

Architecture (gemma3-style, as the GGUF encodes it): sandwich RMSNorm, fused QKV, GQA 16/8,
NEOX half-split RoPE theta=500000, QK-norm, SwiGLU, tied unembedding.

Usage:
  GGUF_PY=/path/to/llama.cpp/gguf-py python3 numpy_forward.py /path/to/afm-ifp3-L35-f16.gguf
"""
import os, sys, numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: [GGUF_PY=/path/to/llama.cpp/gguf-py] numpy_forward.py <model.gguf>")
GGUF = sys.argv[1]
gguf_py = os.environ.get("GGUF_PY")
if gguf_py:
    sys.path.insert(0, gguf_py)
try:
    import gguf
except ImportError:
    sys.exit("need llama.cpp's gguf-py on the path; set GGUF_PY=/path/to/llama.cpp/gguf-py")

NL, D, NH, NKV, HD, EPS, THETA = 35, 1536, 16, 8, 128, 1e-5, 500000.0


def load():
    r = gguf.GGUFReader(GGUF)
    return {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}


def rms(x): return x / np.sqrt((x*x).mean(-1, keepdims=True) + EPS)
def silu(x): return x / (1.0 + np.exp(-x))


def rope(x, pos):                                   # NEOX half-split
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half) / half))
    ang = pos[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)[:, None, :]
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)[:, None, :]
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def forward(ids, T, upto=NL):
    E = T['token_embd.weight']
    h = E[ids] * np.sqrt(D)                          # gemma3 embedding scale
    Tn = len(ids); pos = np.arange(Tn).astype(np.float32)
    for L in range(upto):
        p = f'blk.{L}.'
        x = rms(h) * T[p+'attn_norm.weight']
        q = (x @ T[p+'attn_q.weight'].T).reshape(Tn, NH, HD)
        k = (x @ T[p+'attn_k.weight'].T).reshape(Tn, NKV, HD)
        v = (x @ T[p+'attn_v.weight'].T).reshape(Tn, NKV, HD)
        q = rms(q) * T[p+'attn_q_norm.weight']; k = rms(k) * T[p+'attn_k_norm.weight']
        q = rope(q, pos); k = rope(k, pos)
        o = np.empty((Tn, NH, HD), np.float32); scale = 1.0 / np.sqrt(HD)
        mask = np.triu(np.full((Tn, Tn), -1e30, np.float32), 1)
        for hh in range(NH):
            g = hh // (NH // NKV)
            s = (q[:, hh] @ k[:, g].T) * scale + mask
            s = s - s.max(-1, keepdims=True); e = np.exp(s); a = e / e.sum(-1, keepdims=True)
            o[:, hh] = a @ v[:, g]
        ao = rms(o.reshape(Tn, NH*HD) @ T[p+'attn_output.weight'].T) * T[p+'post_attention_norm.weight']
        h = h + ao
        x = rms(h) * T[p+'ffn_norm.weight']
        ff = (silu(x @ T[p+'ffn_gate.weight'].T) * (x @ T[p+'ffn_up.weight'].T)) @ T[p+'ffn_down.weight'].T
        ff = rms(ff) * T[p+'post_ffw_norm.weight']
        h = h + ff
    h = rms(h) * T['output_norm.weight']
    return h @ E.T                                   # tied unembedding


if __name__ == "__main__":
    print("loading GGUF tensors...", flush=True); T = load()
    ids = [2, 669, 5279, 529, 7001, 563]             # BOS + a sample sequence
    lg = forward(ids, T)
    print("logits:", lg.shape, "finite:", bool(np.isfinite(lg).all()))
    nxt = int(lg[-1].argmax())
    seq = list(ids)
    for _ in range(8):
        seq.append(int(forward(seq, T)[-1].argmax()))
    print("argmax next id:", nxt, "| greedy continuation ids:", seq[len(ids):])
    print("(Expected per paper/: degenerate 'copy' output -- this is the documented negative result.)")
