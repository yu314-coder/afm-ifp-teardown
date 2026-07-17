"""afm_forward_deployed.py -- reconstructed forward for the DEPLOYED ifp1_r48 config.

Assembles everything recovered as of 2026-07-15. Two points are provably blocked from the
shipped assets (marked BLOCKED below); the rest is pinned. This is a faithful scaffold, not a
runnable generator -- see the boundary notes.

DEPLOYED STRUCTURE (from metadata.bin, cross-validated against config.json ifp1_r48):
  - Sparse FFN per layer = 10 active + 4 shared experts; active_ffn_dim = 2560 = 10*256.
    Confirmed: metadata.bin down-proj table (marker 0x0600beef) = 14,080 tiles of 8x1536 4-bit,
    grouped 88 x 160 (col1 = layer-group, col2 = within-group idx 0..9 x16). 88 = 44*2 groups;
    2 groups/layer x 5 experts = 10 active experts = active_ffn_dim (byte-exact match to config).
  - FFN op naming (from MLIR symbol pool): gate = feed_forward_hidden_transform_linear_0,
    up = linear_1, down = feed_forward_output_transform. Ungated permutation-invariant SwiGLU
    sum over resident experts (no per-expert gating in the graph).
  - Attention: 23 standard (full qkv) + 21 kv-reuse (q-only) layers; NKV=4; RoPE theta=500000
    interleaved; QKNorm[128] per head; sandwich norms (attn/ffn pre+post) -- these are explicit
    plain constants in the odix export graph, folded to parameter-free RMSNorm in the compiled
    mpsgraph, so gamma=1 is correct at runtime.
  - Codec: 4-bit index -> 16-entry codebook (afm_codebook_deswz.npy) -> per-1024 fp16 block
    scale -> ANE 8x128 de-swizzle. Down-proj tile = 48x256 z-order (cracked).

BLOCKED (provably, from shipped assets):
  (A) TOKEN EMBEDDING (input lookup + tied unembed): absent from every shipped asset in decodable
      form. It is a CPU-side odix `load_embeddings` op; the table is host/FoundationModels-provided
      and not in this MobileAsset (verified: 4.16M raster offsets + all sibling FM assets, scored
      against a Qwen-calibrated semantic oracle, best GAP << the +0.30 real-embedding bar).
      => `embed(ids)` and `unembed(h)` cannot be filled. Without them the forward has no input
         and no logits, so it cannot be validated end-to-end.
  (B) DEPLOYED EXPERT GATHER: metadata.bin gives the deployed tile *resident-buffer* addresses,
      not raster offsets (decay ~1.1 at every raster mapping). Mapping resident->raster-superset
      (the pruning list = which 10-of-superset per layer) is not in the decoded record fields.
      The raster expert region stores the un-pruned SUPERSET (~219/layer); summing it ungated is
      the UN-pruned base FFN, not the deployed pruned FFN. => `gather_experts(layer)` below returns
      the superset sum as a placeholder; the deployed selection needs the resident->raster map.

Everything except (A) and (B) is validated in the sibling scripts / paper.
"""
import numpy as np

VOCAB = 262144          # Gemma-style 2^18 (confirmed 4 ways; NOT 152064)
D = 1536                # hidden
N_LAYERS = 44
DENSE_FFN = [i for i in range(12)]     # 12 dense-FFN layers (Cout=3072)
ACTIVE_FFN_DIM = 2560   # 10 active experts * 256  (ifp1_r48; from metadata.bin)
SHARED_EXPERTS = 4
EXPERT_SIZE = 256
NKV = 4
ROPE_THETA = 500000.0
HEAD_DIM = 128


def embed(token_ids):
    """BLOCKED (A): token embedding table is host-provided, absent from this MobileAsset."""
    raise NotImplementedError(
        "token embedding unavailable from shipped assets (host/FoundationModels-provided); "
        "see boundary note (A)")


def unembed(hidden):
    """BLOCKED (A): tied to embed(); same table."""
    raise NotImplementedError("tied unembed unavailable; see boundary note (A)")


def gather_deployed_experts(raster, layer, meta):
    """Return the resident-expert gate/up/down for a sparse layer.

    metadata.bin gives 10 active + 4 shared experts/layer with resident-buffer tile addresses.
    BLOCKED (B): those addresses do not map to raster offsets, and the resident->raster
    pruning list is not in the decoded record fields. Callers currently fall back to the
    ungated superset sum from the raster expert region (the UN-pruned base FFN).
    """
    raise NotImplementedError(
        "deployed expert gather blocked: metadata addresses are resident-buffer offsets, "
        "not raster; pruning list (which-of-superset) not recovered -- see boundary note (B)")


def sparse_ffn_ungated(h_norm, gate, up, down):
    """Ungated permutation-invariant SwiGLU over resident experts (structure is validated)."""
    import numpy as np
    def silu(x):
        return x / (1.0 + np.exp(-x))
    inter = silu(h_norm @ gate.T) * (h_norm @ up.T)   # [T, EH]
    return inter @ down                                # [T, D]


# The attention path, RoPE, QKNorm, sandwich-norm placement, and dense FFN are implemented and
# validated in src/afm_forward_working.py; this module documents the DEPLOYED sparse-FFN wiring
# and the two hard boundaries above. A runnable end-to-end forward requires resolving (A) and (B),
# neither of which is recoverable from the shipped GenerativeModels MobileAsset.

if __name__ == "__main__":
    print("Deployed forward scaffold. Pinned:")
    print(f"  VOCAB={VOCAB}, D={D}, layers={N_LAYERS}, NKV={NKV}, rope_theta={ROPE_THETA}")
    print(f"  sparse FFN active_ffn_dim={ACTIVE_FFN_DIM} (10 active + {SHARED_EXPERTS} shared), "
          f"expert_size={EXPERT_SIZE}")
    print("BLOCKED from shipped assets: (A) token embedding [host-provided], "
          "(B) deployed expert gather [resident->raster pruning map].")
