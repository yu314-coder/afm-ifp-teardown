"""Decode afmplus-v11.0-nano's token embedding from program.odix.

LAW (empirically recovered, semantically validated):
    value[t, d] = float(q[t, d]) * scale[d]          # per-dimension symmetric int4, zero-point 0

    nibble(t, d) = BLOB_NIB + (t // 8) * (8 * D) + 8 * d + (t % 8)
    lane = t % 8, D = 2048, LSB-first nibble packing (even index -> low nibble)

  * BLOB byte range = [0x1b78238, 0x11b78238)  == exactly 262144*2048/2 bytes.
    It begins immediately after the "myPlatformId\0\0\0\0" string that terminates the
    odix flatbuffer, and ends exactly where the trailing tensor descriptors start.
  * scale = 2048 fp16 per-DIMENSION scales at file byte 0x11d78338, stored as 64+16+8 = 88
    replicas (runs of 64, 16 and 8 at 0x11d78338 / 0x11db8378 / 0x11dc83b8) -- matching the
    gather_embeddings_{64,16,8} op variants named in the same program.
  * plain [8,1,1] token interleave -- NO lane skew (unlike the stale pico note).
  * The final 144 rows (ids 262000..262143 = <extra_id_262000..>) are exactly zero: the
    untrained tail.  This 147456-byte zero run sits flush against the end of the blob, which
    is what proves there is no lane skew.

NOTE on token ids: tok_vocab.json carries 4 spurious duplicate specials at the head, so
    real_id = json_index - 4.
"""
import numpy as np

ODIX = ('/Volumes/D/github/afm-ifp-teardown/local/nano_asset/model.odixpackage/program.odix')
VOCAB, DIM = 262144, 2048
BLOB      = 0x1b78238          # byte offset of token 0, dim 0
SCALE_OFF = 0x11d78338         # 2048 fp16 per-dimension scales (first of 88 replicas)

_d = np.memmap(ODIX, dtype=np.uint8, mode='r')
SCALE = np.frombuffer(bytes(_d[SCALE_OFF:SCALE_OFF + DIM * 2]), dtype=np.float16).astype(np.float32)
_G = VOCAB // 8                                   # 32768 groups of 8 interleaved tokens
_blob = _d[BLOB:BLOB + VOCAB * DIM // 2].reshape(_G, DIM, 4)   # [group, dim, byte-of-4-lane-pairs]


def codes(groups=None):
    """Signed int4 codes, shape [8*len(groups), DIM]."""
    g = slice(None) if groups is None else groups
    b = np.asarray(_blob[g]).astype(np.int16)                   # [G, DIM, 4]
    n = np.empty(b.shape[:2] + (8,), np.int16)
    n[..., 0::2] = b & 0xF                                      # even lane -> low nibble
    n[..., 1::2] = b >> 4                                       # odd  lane -> high nibble
    n = np.where(n > 7, n - 16, n)                              # signed int4, zero-point 0
    return np.moveaxis(n, 2, 1).reshape(-1, DIM)                # [tok, dim]


def rows(ids):
    """float32 embedding rows for arbitrary token ids."""
    ids = np.atleast_1d(np.asarray(ids, np.int64))
    g, l = ids // 8, ids % 8
    b = np.asarray(_blob[g, :, l // 2]).astype(np.int16)        # [n, DIM]
    q = np.where((l % 2)[:, None].astype(bool), b >> 4, b & 0xF)
    q = np.where(q > 7, q - 16, q)
    return q.astype(np.float32) * SCALE


def table(out_path=None):
    """Full [262144, 2048] float32 table (memmapped to out_path when given)."""
    E = (np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(VOCAB, DIM))
         if out_path else np.empty((VOCAB, DIM), np.float32))
    for s in range(0, _G, 1024):
        e = min(s + 1024, _G)
        E[s * 8:e * 8] = codes(slice(s, e)).astype(np.float32) * SCALE
    return E


if __name__ == '__main__':
    E = table('/Volumes/D/github/afm-ifp-teardown/local/nano_gguf/nano_embedding.npy')
    print('wrote', E.shape, E.dtype)
