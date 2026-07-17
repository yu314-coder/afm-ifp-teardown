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
sys.path.insert(0, "/Volumes/D/fix/pico_shapes")
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
    return picolib.R(W)


def main():
    m = json.load(open(MAP_PATH))
    rep = {"model": "afmplus-v11.0-pico", "note": "decode validation; per-block R (assembly SV-invisible)",
           "tensors": []}
    for e in m:
        if e.get("role") == "PARTIAL_UNIT":
            continue
        W = decode_tensor(e)
        if W is None:
            rep["tensors"].append({"layer": e["layer"], "role": e["role"], "R": None,
                                   "note": "palettized (down) or unresolved"})
            continue
        rep["tensors"].append({"layer": e["layer"], "role": e["role"], "shape": list(W.shape),
                               "R": round(structure_R(W[:256, :512]), 3)})
    ok = [t for t in rep["tensors"] if t.get("R")]
    print("decoded %d/%d tensors; mean per-block R=%.2f (real int4 >~3; scrambled ~1.4)"
          % (len(ok), len(rep["tensors"]), np.mean([t["R"] for t in ok])))
    json.dump(rep, open("/Volumes/D/fix/pico_shapes/pico_decode_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
