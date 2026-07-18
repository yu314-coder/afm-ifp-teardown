"""pico_forward.py -- a RUNNABLE from-weights PyTorch forward for afmplus-v11.0-pico (300M draft).

Assembles the 168 decoded weight tensors (src/pico_weights.py: decode_tensor / decode_down, backed by
the validated picolib codebook decode) into a working 24-layer GQA + SwiGLU transformer and runs it on a
captured host-DRAM embedding row (tw_*.core via embedding_dynamic_capture). Prints the residual-stream
norm at every layer.

=================================================================================================
HONESTY / SCOPE  (read PICO_WEIGHT_RESULT.md + PICO_ARRANGEMENT_RESULT.md first)
=================================================================================================
The weight VALUES are exact (per-tile 16-entry fp16 codebook, R>3 vs ~1 scramble floor, all 168 tensors).
What is NOT proven is the byte-exact intra-tensor ARRANGEMENT: which of a block's 16 tiles lands in which
4x4 cell, the byte order inside each 128x128 tile, block->quadrant placement, and the logical K/V shape.
Those reduce to a fixed ANE conv weight-layout convention that is provably SV-INVISIBLE (SVD is
permutation-invariant) and is NOT shipped as data in binary_0.hwx. So:

  * This module RUNS end-to-end and its plumbing (shapes, GQA fan-out, RoPE, SwiGLU, residual) is correct.
  * Under the DEFAULT (row-major) ARRANGEMENT the numeric outputs are NOT validated as semantically
    correct -- a wrong-but-stable arrangement is NOT a proof. The per-layer norms below are a *plumbing*
    result (it runs, it is finite, it is reproducible), not evidence the convention is the true one.

Every arrangement choice that the structure test cannot see is exposed as a knob on `Arrangement` so a
future functional forward-pass oracle (the only thing that can resolve this) can swap conventions without
touching the transformer code.

The map's per-tensor `shape` field is [in, out] (D-first): Q/O [1024,1024], K/V [1024,256], gate/up
[1024,3200], down [3200,1024]. Weights are therefore used directly as [in,out] with y = x @ W.
"""
import sys, json, math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Volumes/D/fix/afm-ifp-teardown/src")
sys.path.insert(0, "/Volumes/D/fix/pico_shapes")
import pico_weights as pw           # decode_tensor, decode_block, decode_down  (uses picolib)
import picolib                       # raw byte access (_d), codebook helpers
from embedding_dynamic_capture import find_identical_row_buffer

MAP_PATH = pw.MAP_PATH


# =================================================================================================
# Arrangement: every element->position convention the SV structure test cannot arbitrate.
# Defaults are the standard row-major ANE convention (consistent with all proven facts, NOT proven).
# =================================================================================================
@dataclass
class Arrangement:
    # --- architecture (proven from program.odix op graph) ---
    D: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn: int = 3200
    n_layers: int = 24
    vocab: int = 262144
    rope_theta: float = 500000.0
    rope_interleaved: bool = True
    rms_eps: float = 1e-5
    # --- unproven arrangement knobs (SV-invisible; swap for a functional oracle) ---
    tile_grid: tuple = (4, 4)          # N-block: 16 [128,128] tiles -> [512,512] (row-major 4x4)
    kv_mode: str = "reshape"           # single 512x512 K/V block -> [1024,256]:  "reshape" | "topslice"
    s_tile_shape: tuple = (64, 128)    # gate/up 's' half-block: 16 tiles -> [1024,128]
    s_position: str = "append_cols"    # 's' occupies the last 128 output columns of gate/up
    transpose: dict = field(default_factory=dict)   # optional per-role [in,out]->[out,in] flips
    qk_norm: bool = False              # per-head RMSNorm on q,k (odix hints QKNorm[128]; no gamma found)


# =================================================================================================
# Weight assembly (arrangement-aware; values from pico_weights/picolib, geometry from Arrangement)
# =================================================================================================
def _decode_s_block(off_hex, s_tile_shape=(64, 128)):
    """gate/up 's' half-block -> [1024,128] via 16 codebook tiles at stride 0x1080."""
    d = picolib._d
    base = int(off_hex, 16)
    th, tw = s_tile_shape                      # tile element count must be 8192 (=131072/16)
    assert th * tw == 8192, "s tile must hold 8192 int4"
    tl = []
    for t in range(16):
        o = base + t * 0x1080
        cb = np.frombuffer(bytes(d[o:o + 32]), dtype=np.float16).astype(np.float32)  # 16-entry codebook
        r = np.asarray(d[o + 128:o + 128 + 4096])                                    # 8192 int4
        nib = np.empty(8192, np.uint8); nib[0::2] = r & 0xF; nib[1::2] = r >> 4
        tl.append(cb[nib].reshape(th, tw))
    # 16 tiles of [64,128] stacked vertically -> [1024,128]  (last 128 out-columns of the FFN weight)
    return np.vstack(tl)


def _decode_kv(entry, arr: Arrangement):
    """K/V: one physical 512x512 block -> logical [in=1024, out=256] (map shape [1024,256])."""
    blk = pw.decode_block(entry["block_offsets"][0])          # [512,512], codebook-decoded
    if arr.kv_mode == "reshape":
        return blk.reshape(arr.D, 256)                         # 262144 -> [1024,256] row-major
    elif arr.kv_mode == "topslice":
        return blk[:, :256]                                    # [512,256] top slice, zero-padded below
    raise ValueError(arr.kv_mode)


def _decode_gate_up(entry, arr: Arrangement):
    """gate/up: 12 N-blocks (via decode_tensor -> [1024,3072]) + 's' half-block -> [1024,3200]."""
    n_part = pw.decode_tensor(entry)                           # [1024,3072] (decode_tensor drops 's')
    s_part = _decode_s_block(entry["block_offsets"][0], arr.s_tile_shape)  # [1024,128]
    if arr.s_position == "append_cols":
        return np.hstack([n_part, s_part])                    # [1024,3200]
    elif arr.s_position == "prepend_cols":
        return np.hstack([s_part, n_part])
    raise ValueError(arr.s_position)


def load_layer_weights(entry_by_role, arr: Arrangement):
    """Return dict role -> torch.float32 weight in [in,out] orientation for one layer."""
    W = {}
    W["Q"] = pw.decode_tensor(entry_by_role["Q"])             # [1024,1024]
    W["O"] = pw.decode_tensor(entry_by_role["O"])             # [1024,1024]
    W["K"] = _decode_kv(entry_by_role["K"], arr)              # [1024,256]
    W["V"] = _decode_kv(entry_by_role["V"], arr)              # [1024,256]
    W["gate"] = _decode_gate_up(entry_by_role["gate"], arr)   # [1024,3200]
    W["up"] = _decode_gate_up(entry_by_role["up"], arr)       # [1024,3200]
    W["down"] = pw.decode_down(entry_by_role["down"])         # [3200,1024]
    out = {}
    for r, m in W.items():
        m = np.ascontiguousarray(m, dtype=np.float32)
        t = torch.from_numpy(m)
        if arr.transpose.get(r):
            t = t.T.contiguous()
        out[r] = t
    return out


def load_all_weights(arr: Arrangement, map_path=MAP_PATH):
    m = json.load(open(map_path))
    by_layer = {}
    for e in m:
        if e.get("role") == "PARTIAL_UNIT":
            continue
        by_layer.setdefault(e["layer"], {})[e["role"]] = e
    layers = []
    for li in range(arr.n_layers):
        layers.append(load_layer_weights(by_layer[li], arr))
    return layers


# =================================================================================================
# Forward primitives
# =================================================================================================
def rmsnorm(x, eps, gamma=None):
    """Parameter-free RMSNorm (no learned gamma recovered; gamma optional if a norms task finds it)."""
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return n if gamma is None else n * gamma


def rope(x, theta, interleaved=True):
    """x: [T, H, head_dim]. Interleaved RoPE (pairs (2i,2i+1)), theta base = 500000."""
    T, H, hd = x.shape
    half = hd // 2
    pos = torch.arange(T, dtype=torch.float32)[:, None]                      # [T,1]
    inv = theta ** (-torch.arange(0, half, dtype=torch.float32) / half)      # [half]
    ang = pos * inv[None, :]                                                  # [T,half]
    cos = torch.cos(ang)[:, None, :]; sin = torch.sin(ang)[:, None, :]       # [T,1,half]
    if interleaved:
        xe = x[..., 0::2]; xo = x[..., 1::2]                                 # even/odd
        ye = xe * cos - xo * sin
        yo = xe * sin + xo * cos
        y = torch.empty_like(x); y[..., 0::2] = ye; y[..., 1::2] = yo
        return y
    xe = x[..., :half]; xo = x[..., half:]
    return torch.cat([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1)


def attention(x_norm, W, arr: Arrangement):
    T = x_norm.shape[0]
    nq, nkv, hd = arr.n_heads, arr.n_kv_heads, arr.head_dim
    q = (x_norm @ W["Q"]).view(T, nq, hd)                    # [T,16,64]
    k = (x_norm @ W["K"]).view(T, nkv, hd)                   # [T,4,64]
    v = (x_norm @ W["V"]).view(T, nkv, hd)                   # [T,4,64]
    if arr.qk_norm:
        q = rmsnorm(q, arr.rms_eps); k = rmsnorm(k, arr.rms_eps)
    q = rope(q, arr.rope_theta, arr.rope_interleaved)
    k = rope(k, arr.rope_theta, arr.rope_interleaved)
    g = nq // nkv                                            # GQA group size = 4
    k = k.repeat_interleave(g, dim=1)                        # [T,16,64]
    v = v.repeat_interleave(g, dim=1)
    q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)   # [H,T,64]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)      # [H,T,T]
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    scores = scores + mask
    a = torch.softmax(scores, dim=-1) @ v                   # [H,T,64]
    a = a.transpose(0, 1).reshape(T, nq * hd)               # [T,1024]
    return a @ W["O"]                                       # [T,1024]


def ffn(x_norm, W):
    g = x_norm @ W["gate"]                                   # [T,3200]
    u = x_norm @ W["up"]                                     # [T,3200]
    return (F.silu(g) * u) @ W["down"]                       # [T,1024]


def forward(emb, layers, arr: Arrangement, verbose=True):
    """emb: [T, D] float32 residual stream (captured embedding rows). Returns final hidden, prints norms."""
    x = emb.clone()
    per_layer = []
    if verbose:
        print("  input     residual |x| = %.4f   (T=%d, D=%d)" % (x.norm().item(), x.shape[0], x.shape[1]))
    for li, W in enumerate(layers):
        x = x + attention(rmsnorm(x, arr.rms_eps), W, arr)
        x = x + ffn(rmsnorm(x, arr.rms_eps), W)
        nrm = x.norm().item()
        rms = x.pow(2).mean().sqrt().item()
        per_layer.append(nrm)
        if verbose:
            print("  layer %2d  residual |x| = %10.4f   rms = %8.5f" % (li, nrm, rms))
    x = rmsnorm(x, arr.rms_eps)         # final norm (parameter-free)
    return x, per_layer


# =================================================================================================
def load_embedding(core_path, D=1024, seq_len=1):
    row, n, off = find_identical_row_buffer(core_path, D=D)
    if row is None:
        raise RuntimeError("no captured embedding row in %s" % core_path)
    v = torch.from_numpy(row.astype(np.float32))
    return v.unsqueeze(0).repeat(seq_len, 1), n, off      # [seq_len, D]


def main():
    core = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/D/fix/tw_dog.core"
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    arr = Arrangement()
    print("=== afmplus-v11.0-pico  from-weights forward  (DEFAULT row-major arrangement) ===")
    print("arch: %d layers  D=%d  %dQ/%dKV heads  head_dim=%d  ffn=%d  RoPE theta=%g interleaved=%s"
          % (arr.n_layers, arr.D, arr.n_heads, arr.n_kv_heads, arr.head_dim, arr.ffn,
             arr.rope_theta, arr.rope_interleaved))
    print("loading + assembling 24x7 weight tensors from binary_0.hwx ...")
    layers = load_all_weights(arr)
    # sanity: report assembled shapes for layer 0
    s0 = {r: tuple(t.shape) for r, t in layers[0].items()}
    print("layer-0 weight shapes [in,out]:", s0)
    emb, n, off = load_embedding(core, D=arr.D, seq_len=seq_len)
    print("embedding: %s  from %s  (%d identical rows @0x%x, nnz=%d)"
          % (tuple(emb.shape), core.split("/")[-1], n, off, int((emb[0] != 0).sum())))
    print("--- running forward, residual-stream norm per layer ---")
    with torch.no_grad():
        x, per_layer = forward(emb, layers, arr)
    print("final (post-norm) hidden |x| = %.4f   shape=%s" % (x.norm().item(), tuple(x.shape)))
    print("OK: forward ran end-to-end over %d layers." % arr.n_layers)
    return per_layer


if __name__ == "__main__":
    main()
