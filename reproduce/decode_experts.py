#!/usr/bin/env python3
"""decode_experts.py -- decode the ifp sparse-expert pool from the model you already have.

RESEARCH / INTEROPERABILITY USE ONLY. This script contains NO Apple data. It reads the
`ifp_rasterized_weights.bin` that already exists inside the Foundation Models asset on YOUR OWN
Apple device and decodes it to floating-point expert tensors in memory. It redistributes nothing:
Apple's weights stay on your machine. You are responsible for using it consistently with your
local law and Apple's software license.

Container (offsets verified against the file's own header; byte-exact):
  0x00  'BBBB'                              magic
  0x50  4 x fp16 palette = [-1.5,-0.5,0.5,1.5]
  0x60  DOWN per-row scales : L x 1536 fp16
  0x21060 GATE/UP per-row scales : L x E x 512 fp16
  0x1078000 payload: 2-bit codes, 4 per byte, LSB-first
Payload geometry (parameter-free): experts = payload_codes / (3*256*1536) = L x E.
  expert block = gate[256,1536] | up[256,1536] | down[1536,256]
  every matrix = (nout/16) banks of 16 output rows; within a bank the 2-bit codes are
  i-major with o fastest:  code index p -> (out = 16*bank + p%16, in = p//16).

Usage:
  python3 decode_experts.py /path/to/model.odixpackage/ifp/ifp_rasterized_weights.bin
Find the asset with:  reproduce/find_asset.sh
"""
import sys, numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: decode_experts.py <path to ifp_rasterized_weights.bin>")
RW = sys.argv[1]

L, E, H, ES = 44, 380, 1536, 256          # layers, stored experts/layer, hidden, expert_size
MATB = ES * H // 4                         # 98304 bytes per matrix (2-bit)
EXPB = 3 * MATB                            # gate|up|down
LAYB = E * EXPB
SB_OFF, SB_SZ = 0x60, 0x21000
SC_OFF = 0x21060
PD_OFF = 0x1078000

_mm = np.memmap(RW, dtype=np.uint8, mode='r')
assert _mm[:4].tobytes() == b'BBBB', "not an ifp_rasterized_weights.bin (bad magic)"
PAL = np.frombuffer(_mm[0x50:0x58].tobytes(), dtype='<f2').astype(np.float32)
SC_DN = np.frombuffer(_mm[SB_OFF:SB_OFF+SB_SZ].tobytes(), dtype='<f2').astype(np.float32).reshape(L, H)
SC_GU = np.memmap(RW, dtype='<f2', mode='r', offset=SC_OFF, shape=(L, E, 512))


def _codes(off, nbytes):
    b = np.frombuffer(_mm[off:off+nbytes].tobytes(), dtype=np.uint8)
    z = np.empty((b.size, 4), np.uint8)
    z[:, 0] = b & 3; z[:, 1] = (b >> 2) & 3; z[:, 2] = (b >> 4) & 3; z[:, 3] = (b >> 6) & 3
    return z.reshape(-1)


def expert(layer, exp, which):
    """which in {0:gate, 1:up, 2:down} -> float32 [256,1536] (gate/up) or [1536,256] (down)."""
    no, ni = (ES, H) if which < 2 else (H, ES)
    c = _codes(PD_OFF + layer*LAYB + exp*EXPB + which*MATB, MATB)
    W = PAL[c.reshape(no//16, ni, 16).transpose(0, 2, 1).reshape(no, ni)]
    s = SC_DN[layer] if which == 2 else np.asarray(SC_GU[layer, exp, which*256:(which+1)*256], np.float32)
    return W * s[:, None]


if __name__ == "__main__":
    print("palette:", PAL, "| file magic OK | L=%d E=%d" % (L, E))
    for w, nm in [(0, "gate"), (1, "up"), (2, "down")]:
        A = expert(0, 0, w)
        print("  %-5s %s  mean %.3e  sd %.4e  absmax %.3f" % (nm, A.shape, A.mean(), A.std(), np.abs(A).max()))
    A = expert(L-1, E-1, 2)
    print("  coverage check L%d e%d down: sd %.4e" % (L-1, E-1, A.std()))
    print("OK: expert pool decodes. Import `expert(layer, exp, which)` to use.")
