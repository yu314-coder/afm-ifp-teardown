"""picolib.py -- validated pico weight decode helpers (shared by the overnight workflow).
Boundaries from LC_SYMTAB (K<hash>_ne_<0..15>); tile = 0x2080 (128B fp16-scale hdr + 128x128 int4);
value = (nibble - 7.5); arrange 16 tiles into a grid. Structure test R: real int4 weight ~3.6-6, scrambled ~1.4.
"""
import struct, numpy as np, numpy.linalg as la

HWX = "/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/031c7be6f8fddbff0a6650fee75e345b1ee9613c.asset/.AssetData/model.odixpackage/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx"
_raw = open(HWX, "rb").read()
_nc = struct.unpack("<I", _raw[16:20])[0]
_o = 32
symoff = nsyms = stroff = 0
for _ in range(_nc):
    _c, _cz = struct.unpack("<II", _raw[_o:_o + 8])
    if _c == 0x2:
        symoff, nsyms, stroff, _ss = struct.unpack("<4I", _raw[_o + 8:_o + 24])
    _o += _cz

# n_sect -> (vmaddr base, file base) for the three weight segments
SEG = {8: (0x394d8000, 0x22c4000), 9: (0x414d4000, 0xa2c0000), 10: (0x42570000, 0xb35c000)}
SEGNAME = {8: "__kern_0", 9: "__kern_1", 10: "__kern_2"}

# TENS = sorted list of (vaddr, n_sect) for each weight tensor (its _ne_0 tile base)
TENS = []
_nul = _raw.find(b"\x00", 0)  # noop to keep b"\x00" literal intact
for _i in range(nsyms):
    _b = _raw[symoff + _i * 16: symoff + _i * 16 + 16]
    _strx = struct.unpack("<I", _b[:4])[0]
    _ns = _b[5]
    _nv = struct.unpack("<Q", _b[8:16])[0]
    if _ns not in SEG:
        continue
    _e = _raw.index(b"\x00", stroff + _strx)
    _nm = _raw[stroff + _strx:_e].decode("latin1")
    if _nm.endswith("_ne_0"):
        TENS.append((_nv, _ns))
TENS.sort()

_d = np.memmap(HWX, dtype=np.uint8, mode="r")


def foff(va, ns):
    vb, fb = SEG[ns]
    return va - vb + fb


def tiles(va, ns, hdr=0):
    """16 int4 tiles, each transposed [128,128]. hdr=bytes skipped at tile start (0 tested best)."""
    base = foff(va, ns)
    T = []
    for t in range(16):
        oo = base + t * 0x2080 + hdr
        r = np.asarray(_d[oo:oo + 8192]).astype(np.int16)
        q = np.empty(16384, np.float32)
        q[0::2] = r & 0xF
        q[1::2] = r >> 4
        T.append((q - 7.5).reshape(128, 128).T)
    return T


def arrange(T, gr, gc):
    return np.block([[T[r * gc + c] for c in range(gc)] for r in range(gr)])


def smax(M):
    M = M.astype(np.float32)
    v = np.random.RandomState(1).randn(M.shape[1]); v /= la.norm(v)
    for _ in range(4):
        x = M @ v; x /= la.norm(x) + 1e-9
        v = M.T @ x; v /= la.norm(v) + 1e-9
    return la.norm(M @ v)


def R(W):
    W = W.astype(np.float32)
    s1 = smax(W)
    f = W.flatten().copy(); np.random.default_rng(1).shuffle(f)
    return float((s1 / (smax(f.reshape(W.shape)) + 1e-9)) ** 2)


def best_shape(va, ns):
    """Return (bestR, 'HxW', gr, gc) over candidate 16-tile arrangements."""
    T = tiles(va, ns)
    best = (0.0, None, 0, 0)
    for gr, gc in [(4, 4), (8, 2), (2, 8), (16, 1), (1, 16)]:
        r = R(arrange(T, gr, gc))
        if r > best[0]:
            best = (r, "%dx%d" % (gr * 128, gc * 128), gr, gc)
    return best


if __name__ == "__main__":
    print("weight tensors:", len(TENS),
          "kern0:", sum(1 for _, s in TENS if s == 8),
          "kern1:", sum(1 for _, s in TENS if s == 9),
          "kern2:", sum(1 for _, s in TENS if s == 10))


# --- 2026-07-18 correctness fix (verified): per-tile 16-entry fp16 codebook, not linear (q-7.5)*S ---
# The 128-byte tile header begins with a monotonic non-uniform 16-entry fp16 codebook; the true weight
# read is W = codebook[nibble] (linear (q-7.5)*S leaves 14-19% residual). Geometry conclusions unchanged
# (dequant is element-wise, arrangement-orthogonal).
def tile_codebook(va, ns, t):
    o = foff(va, ns) + t * 0x2080
    return np.frombuffer(bytes(_d[o:o + 32]), dtype=np.float16).astype(np.float32)  # 16 entries

def tiles_cb(va, ns):
    """16 tiles decoded fully: W = codebook[nibble] * per-8row-group scale (grouped-palettized).
    Header: bytes[0:32]=16-entry fp16 codebook, bytes[64:96]=16 fp16 per-group scales (one per 8 rows).
    Codebook-only was ~9x over-scaled; the per-group scale gives real transformer magnitudes (rms~0.03)."""
    base = foff(va, ns); out = []
    for t in range(16):
        o = base + t * 0x2080
        cb = np.frombuffer(bytes(_d[o:o + 32]), dtype=np.float16).astype(np.float32)       # codebook
        sc = np.frombuffer(bytes(_d[o + 64:o + 96]), dtype=np.float16).astype(np.float32)   # per-group scale
        r = np.asarray(_d[o + 128: o + 128 + 8192])
        nib = np.empty(16384, np.uint8); nib[0::2] = r & 0xF; nib[1::2] = r >> 4
        W = cb[nib].reshape(128, 128) * np.repeat(sc, 8)[:128][:, None]  # scale per 8-row group
        out.append(W.T)
    return out
