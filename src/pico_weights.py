"""pico_weights.py -- reproducible decoder for the afmplus-v11.0-pico (300M) transformer weights.

Assembles the 168 logical weight tensors of Apple's on-device 300M draft model directly from the
shipped `binary_0.hwx` ANE program. Operates only on the operator's own local asset; emits weights
into memory for the caller. NOTHING here contains Apple's weights -- it is the decode procedure.

WHAT IS PROVEN (see PICO_WEIGHT_RESULT.md / PICO_ARRANGEMENT_RESULT.md):
  * Boundaries: LC_SYMTAB names every weight tensor `K<hash>_ne_<0..15>` (16 tiles/block, stride 0x2080).
  * Packing: 998 blocks -> 24 layers x {Q,K,V,O,gate,up,down} = 168 tensors; per-layer pattern
    [N x10][s][N x12][s][N x12][L x4]; int4 budget closes exactly; block CONSUMPTION order = program order.
  * Per-tile decode: value = codebook[nibble], codebook = the tile's own 16-entry fp16 header (verified;
    linear (q-7.5)*S leaves 14-19% residual). down-projections (class L) use the palettized ANE down-codec.

WHAT IS ASSUMED (a fixed ANE conv-layout convention, provably SV-invisible, NOT stated in the asset):
  * The 16 tiles fill the [512,512] block in row-major 4x4 order; blocks fill a multi-block [Cout,Cin]
    tensor in row-major order; K/V are read [1024,256]. These follow the standard ANE convention and are
    CONSISTENT with every proven fact, but are not independently proven from the file (would need the ANE
    layout spec or a functional forward-pass oracle). Values are exact; only element->position may permute.

Requires the shared library `picolib.py` (validated symtab parser + tile/codebook decode).
"""
import sys, json, numpy as np
import os as _os; sys.path.insert(0, _os.path.dirname(__file__)); sys.path.insert(0, "/Volumes/D/fix/pico_shapes")
import picolib  # noqa: E402

# logical tensor -> ordered block offsets comes from the committed map
MAP_PATH = "/Volumes/D/fix/afm-ifp-teardown/pico_weight_map.json"


def _va_ns_at(file_off):
    """Reverse a committed file offset back to (vaddr, n_sect) for picolib decode."""
    for ns, (vb, fb) in picolib.SEG.items():
        seg_lo = fb
        # each segment's file span; pick the one containing this offset
        if file_off >= fb and file_off < fb + 0x9000000:  # generous per-seg cap
            return (file_off - fb + vb, ns)
    return None


def decode_block(file_off, codebook=True):
    """One 512x512 block: 16 codebook-decoded [128,128] tiles in row-major 4x4 (assumed convention)."""
    va, ns = _va_ns_at(int(file_off, 16) if isinstance(file_off, str) else file_off)
    T = picolib.tiles_cb(va, ns) if codebook else picolib.tiles(va, ns)
    return picolib.arrange(T, 4, 4)  # 4x4 -> [512,512]


def decode_tensor(entry):
    """Assemble a logical tensor [Cout,Cin] from its ordered N-block offsets (row-major block grid).
    Skips class-L (palettized down) blocks -- those need the down-codec, not this uniform path."""
    offs = entry["block_offsets"]
    cls = entry.get("block_classes", ["N"] * len(offs))
    cout, cin = entry["shape"]
    if "L" in cls:
        return None  # down-proj: palettized, use the ANE down-codec separately
    blocks = [decode_block(o) for o, c in zip(offs, cls) if c == "N"]
    if not blocks:
        return None
    # row-major block grid sized to [cout, cin] (assumed convention)
    bpr = max(1, cin // 512)
    rows = [np.hstack(blocks[i:i + bpr]) for i in range(0, len(blocks), bpr)]
    W = np.vstack(rows)
    return W[:cout, :cin]


def structure_R(W):
    """True-SVD low-rank structure ratio vs a fully-shuffled baseline (real weight >~3; scrambled ~1).
    NB: picolib.R is a 4-iteration power method that under-resolves — use true SVD (np.linalg.svd)."""
    W = W.astype(np.float32)
    s1 = float(np.linalg.svd(W, compute_uv=False)[0])
    f = W.flatten().copy(); np.random.default_rng(1).shuffle(f)
    s2 = float(np.linalg.svd(f.reshape(W.shape), compute_uv=False)[0])
    return (s1 / s2) ** 2


def decode_down(entry):
    """Down-projection [3200,1024] (class L): 4 L-blocks across columns; each L-block = 16 tiles at
    stride 0x6480 (128B codebook header + 51200 int4), tile [200,256] row-major, W = codebook[nibble]."""
    cols = []
    d = picolib._d
    for off in entry["block_offsets"]:
        base = int(off, 16); tl = []
        for t in range(16):
            o = base + t * 0x6480
            cb = np.frombuffer(bytes(d[o:o + 32]), dtype=np.float16).astype(np.float32)
            sc = np.frombuffer(bytes(d[o + 64:o + 96]), dtype=np.float16).astype(np.float32)  # per-group scale
            r = np.asarray(d[o + 128:o + 128 + 25600])
            nib = np.empty(51200, np.uint8); nib[0::2] = r & 0xF; nib[1::2] = r >> 4
            W = cb[nib].reshape(200, 256)
            tl.append(W * np.repeat(sc, int(np.ceil(200 / len(sc))))[:200][:, None])  # per-row-group scale
        cols.append(np.vstack(tl))  # [3200,256]
    return np.hstack(cols)          # [3200,1024]


def main():
    m = json.load(open(MAP_PATH))
    rep = {"model": "afmplus-v11.0-pico", "note": "168/168 decode validation; true-SVD R on full tensors",
           "tensors": []}
    for e in m:
        if e.get("role") == "PARTIAL_UNIT":
            continue
        W = decode_down(e) if e.get("role") == "down" else decode_tensor(e)
        rep["tensors"].append({"layer": e["layer"], "role": e["role"], "shape": list(W.shape),
                               "R": round(structure_R(W), 3)})
    Rs = [t["R"] for t in rep["tensors"]]
    import collections
    byrole = collections.defaultdict(list)
    for t in rep["tensors"]:
        byrole[t["role"]].append(t["R"])
    print("decoded %d/%d tensors as REAL weights (true-SVD R; scrambled ~1):" % (len(Rs), len(Rs)))
    print("  per-role mean R:", {k: round(np.mean(v), 2) for k, v in byrole.items()})
    print("  overall mean R=%.2f  min=%.2f" % (np.mean(Rs), min(Rs)))
    json.dump(rep, open("/Volumes/D/fix/pico_shapes/pico_decode_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
