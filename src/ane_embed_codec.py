"""ane_embed_codec.py -- decode an ANE palettized [512,1536] conv-weight chunk.

Recovered 2026-07-15 by coreml2hwx positional-probe round-trip (Phase 1 of ANE_DECODE_PLAN.md).
The AFM token-embedding / logit-projection is stored as 512 chunks of a [512,1536] 4-bit
`PalettizedConv2D` weight (odix chunk = 393216 bytes, dtype 0x40012; §find:embed).

The ANE coefficient tiling for a G=1 [Cout=512, Cin=1536] palettized conv is the nested tile

    p(o,i) = (o // 32) * 49152 + (i // 16) * 512 + (i % 16) * 32 + (o % 32)

i.e. innermost 32 output channels, then 16 input channels (stride 32), then 96 input
blocks (stride 512), then 16 output blocks (stride 49152). Validated end-to-end against a
known-weight round-trip: corr(decoded, W) = +0.984 (4-bit-quantization-limited); the
no-tiling control scores 0.002.

CAVEAT: this is the tiling coreml2hwx (coremltools 9.0) emits; it is assumed identical to the
shipped embedding's (same espresso pipeline, same hwx magic/section structure). Confirm against
the shipped data with the semantic oracle (Phase 2) before trusting a recovered embedding.
"""
import numpy as np

COUT, CIN = 512, 1536


def physical_index(cout=COUT, cin=CIN):
    """Return P[o,i] = physical nibble position holding logical weight (o,i)."""
    o = np.arange(cout)[:, None]
    i = np.arange(cin)[None, :]
    return (o // 32) * 49152 + (i // 16) * 512 + (i % 16) * 32 + (o % 32)


def read_kern0_nibbles(hwx_bytes, k0_fileoff, k0_size, nbank=16, hdr=64):
    """Extract the Cout*Cin coefficient nibbles from a __kern_0 section in physical order.

    Banks: `nbank` equal banks, each prefixed by an `hdr`-byte header; within a bank the
    nibbles are low-then-high per byte.
    """
    blob = np.frombuffer(hwx_bytes[k0_fileoff:k0_fileoff + k0_size], dtype=np.uint8)
    bank = k0_size // nbank
    out = []
    for b in range(nbank):
        raw = blob[b * bank + hdr:(b + 1) * bank]
        inter = np.empty(raw.size * 2, np.uint8)
        inter[0::2] = raw & 0x0F
        inter[1::2] = raw >> 4
        out.append(inter)
    return np.concatenate(out)[:COUT * CIN]


def decode_chunk(nibbles_physical, lut):
    """nibbles_physical: uint8[Cout*Cin] in physical order; lut: float[16].
    Returns the [Cout, Cin] dequantized weight."""
    P = physical_index()
    idx = nibbles_physical[P]                      # [Cout, Cin] palette indices
    return np.asarray(lut, np.float32)[idx]        # [Cout, Cin] weights


def linear_lut(wmin, wmax):
    """The linear_lut palette coreml2hwx emits: 16 evenly-spaced levels over [wmin, wmax]."""
    return np.linspace(wmin, wmax, 16).astype(np.float32)


if __name__ == "__main__":
    # Self-test: the closed form must be a bijection over [0, Cout*Cin).
    P = physical_index().ravel()
    assert len(np.unique(P)) == COUT * CIN, "tiling is not a bijection"
    assert P.min() == 0 and P.max() == COUT * CIN - 1
    print(f"OK: physical_index is a bijection over {COUT}x{CIN} = {COUT*CIN} positions")
    print(f"    p(0,0)={physical_index()[0,0]}  p(1,0)={physical_index()[1,0]}  "
          f"p(0,1)={physical_index()[0,1]}  p(31,0)={physical_index()[31,0]}  "
          f"p(32,0)={physical_index()[32,0]}")
