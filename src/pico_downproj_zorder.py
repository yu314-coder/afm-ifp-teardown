"""pico down-projection ANE coefficient z-order, recovered by positional read.

Ground truth obtained by compiling probe convs (coreml2hwx) whose 4-bit weight index at (o,i)
encodes a base-16 digit of o or i, then decoding the digits at every physical nibble position.
Verified: PERFECT BIJECTION over all 256*3200 = 819200 positions.

pico's shipped down-proj ANE task (read from its own binary_0.hwx, Task 218):
    InDim W=64 H=1 C=3200      OutDim W=64 H=1 C=256
    OCGSize=4  ActiveNE=4  SmSrc=1  OutTrans=1   KernelCfg Fmt=FLOAT16 Pal=1(4bit)
    16 coefficient banks, CoeffSize 0x6480 each
So down [3200 -> 1024] is 4 ANE tasks of Cout=256; each task has 16 banks;
each bank = 16 output channels x all 3200 inputs = 51200 nibbles.

CAVEAT: the probe compiles at OutTrans=0 and emits a 64-byte header (0x6440) with no scale
table, while pico ships OutTrans=1 and a 128-byte header (0x6480) carrying 16 scales. Applying
the map below to pico's shipped weights does NOT yet yield a functional improvement, so the
remaining deltas are believed to matter. See PICO_POSREAD_RESULT.md.
"""
import numpy as np

BANK_STRIDE = 0x6480      # pico (probe build is 0x6440)
HEADER      = 128         # pico (probe build is 64)
PAYLOAD     = 25600       # bytes; 51200 nibbles either way
SLOTS       = PAYLOAD * 2

def slot_to_oi(slot, bank):
    """Physical slot within a bank -> (output channel within task, input/neuron index).

    Closed form, exact for all 16 banks:
        o = 16*bank + (slot % 16)      # 16 outputs vary FASTEST
        i = slot // 16
    Note: no input pair swap here, unlike the 3B OCG=4 tile whose input column is ig^1.
    """
    return 16 * bank + (slot % 16), slot // 16

def decode_block(mem, base, task, out=None):
    """Decode one pico down block (16 banks) into [3200, 1024]; task in 0..3."""
    if out is None:
        out = np.zeros((3200, 1024), np.float32)
    slot = np.arange(SLOTS)
    o_loc, neur = slot % 16, slot // 16
    for b in range(16):
        p = base + b * BANK_STRIDE
        cb = np.frombuffer(bytes(mem[p:p + 32]), dtype=np.float16).astype(np.float32)
        sc = np.frombuffer(bytes(mem[p + 64:p + 96]), dtype=np.float16).astype(np.float32)
        pay = np.asarray(mem[p + HEADER:p + HEADER + PAYLOAD])
        nb = np.empty(SLOTS, np.uint8)
        nb[0::2] = pay & 0xF
        nb[1::2] = pay >> 4
        out[neur, task * 256 + 16 * b + o_loc] = cb[nb] * sc[o_loc]   # scale axis UNCONFIRMED
    return out
