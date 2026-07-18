"""pico_embedding.py -- decode the FULL token embedding of Apple's on-device 300M model.

Recovers the complete [262144, 1024] token-embedding table of `afmplus-v11.0-pico` directly from the
shipped CPU program `program.odix`. Because the model's output projection is TIED to the embedding,
this is also the unembed: `logits = h @ E.T`.

NOTHING here contains Apple's weights -- this is the decode procedure. Run it on your own device's asset.

--------------------------------------------------------------------------------------------------
THE LAW (cracked 2026-07-18; validated bit-exactly against dynamically-captured ground truth)
--------------------------------------------------------------------------------------------------
    value[t, d] = float(q[t, d]) * scale[d]        # per-channel symmetric int4, zero-point 0

    nibble(t, d) = NIB0 + (t // 8) * 8192 + 8 * d + lane + SKEW[lane]
    lane = t % 8 ,  SKEW = [0, 0, 0, 0, 8184, 8184, 8184, 8184]
    NIB0 = 1_203_476                                # nibble index of token 0, dim 0

  * The base layout is the `[8,1,1]` token-interleave Apple uses for its small models (8 tokens
    interleaved per dim-major group of 8192 nibbles) -- BUT lanes 4..7 carry a +8184 SKEW. That single
    term is why eight prior layout sweeps (contiguous / interleave-8,16,64 / ANE-tile / palettized /
    affine / scale-table) all read as noise: without it every lane-4..7 token decodes scrambled.
  * `scale`: 1024 fp16 per-DIMENSION scales at file byte 0x8193F48, stored as 88 replicas in runs of
    64/16/8 -- matching the `gather_embeddings_{64,16,8}` op variants in the same program.
  * q is a SIGNED 4-bit code (0..15 -> subtract 16 when > 7); there is no per-token scale and no codebook.

VALIDATION
  * Bit-exact (`np.array_equal`) reproduction of dynamically-captured rows for ▁dog, ▁dogs, ▁king,
    ▁kings, ▁london -- captured independently from the live model via lldb (see embedding_dynamic_capture.py).
  * Whole-table semantics: mean cosine of orthographic pairs +0.588 vs +0.256 random control;
    nearest neighbours of ▁Paris = Dublin, Prague, Munich, Portugal, Brussels, Seattle.
  * Structure: 140 contiguous all-zero rows at ids 262004..262143 (untrained tail), coherent
    cross-lingual neighbours (每天 -> 每日/毎日).
  NB: 5 ground-truth rows are NOT sufficient on their own -- they cover only 3 of the 8 lanes, and a
  skew-less variant reproduces all 5 yet is wrong for ~65k tokens. Always validate whole-table.
"""
import numpy as np

ODIX = ("/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/"
        "031c7be6f8fddbff0a6650fee75e345b1ee9613c.asset/.AssetData/model.odixpackage/program.odix")

VOCAB, DIM = 262144, 1024
NIB0 = 1_203_476
SCALE_OFF = 0x8193F48
SKEW = np.array([0, 0, 0, 0, 8184, 8184, 8184, 8184], dtype=np.int64)

_d = np.memmap(ODIX, dtype=np.uint8, mode="r")
SCALE = np.frombuffer(bytes(_d[SCALE_OFF:SCALE_OFF + DIM * 2]), dtype=np.float16).astype(np.float32)
_dims = np.arange(DIM, dtype=np.int64)


def embed_row(t):
    """Exact embedding vector [1024] float32 for token id t."""
    lane = t % 8
    nib = NIB0 + (t // 8) * 8192 + 8 * _dims + lane + SKEW[lane]
    b = np.asarray(_d[nib >> 1]).astype(np.int16)
    q = np.where((nib & 1).astype(bool), b >> 4, b & 0xF).astype(np.int16)
    q = np.where(q > 7, q - 16, q)          # signed int4, zero-point 0
    return q.astype(np.float32) * SCALE


def embed_table(ids=None):
    """Full [V,1024] table (or just `ids`). The tied unembed is this same matrix: logits = h @ E.T."""
    ids = range(VOCAB) if ids is None else ids
    return np.stack([embed_row(int(t)) for t in ids])


if __name__ == "__main__":
    import json, sys
    vocab = json.load(open("/Volumes/D/fix/afm_odix/tok_vocab.json"))
    idx = {t: i for i, t in enumerate(vocab)}
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    for a, b in [("▁dog", "▁dogs"), ("▁king", "▁kings"), ("▁book", "▁books")]:
        print("  %-8s ~ %-9s cos=%+.3f" % (a, b, cos(embed_row(idx[a]), embed_row(idx[b]))))
