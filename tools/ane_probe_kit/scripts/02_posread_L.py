"""Positional read at pico's L-class geometry (Cin=3200 -> Cout=256, 4-bit palettized).

This is the method that cracked the N-class z-order: compile a conv whose 4-bit weight VALUES
encode the position digits, read the emitted coefficient stream, and invert to get the exact
slot -> (output, input) map. It produces an answer rather than accepting/rejecting a guess.

Probes: o in [0,256) needs 2 base-16 digits, i in [0,3200) needs 3 (16^3 = 4096 > 3200).
    o0 = o % 16      o1 = (o // 16) % 16
    i0 = i % 16      i1 = (i // 16) % 16     i2 = (i // 256) % 16
Each probe tensor is W[o, i] = that digit, so every weight is already an integer in [0,16) and
4-bit palettization recovers it exactly (verified per bank by decoding through the bank's own
codebook -- index != value in general, which was a real bug the first time this was done).

Coefficient bytes are located structurally: each bank is [header][payload], and banks are found
by scanning for the emitted codebook. Everything runs through the working
mlprogram -> coremlcompiler -> mil_to_hwx path under anevenv.
"""
import os, sys, glob, subprocess
import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.converters.mil import Builder as mb

WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'posL')
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'), exist_ok=True)
os.chdir(WORK)
BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin')
HP = os.path.join(BIN, 'hwx_parsing')
MIL2HWX = os.path.join(BIN, 'mil_to_hwx')

CIN, COUT, S = 3200, 256, 64
NBANK = 16
OPB = COUT // NBANK              # 16 outputs per bank
PAYLOAD_NIB = OPB * CIN          # 51200 nibbles per bank
PAYLOAD_B = PAYLOAD_NIB // 2     # 25600 bytes

o_idx = np.arange(COUT)[:, None]
i_idx = np.arange(CIN)[None, :]
PROBES = {
    'o0': (o_idx % 16) * np.ones((1, CIN), np.int64),
    'o1': ((o_idx // 16) % 16) * np.ones((1, CIN), np.int64),
    'i0': np.ones((COUT, 1), np.int64) * (i_idx % 16),
    'i1': np.ones((COUT, 1), np.int64) * ((i_idx // 16) % 16),
    'i2': np.ones((COUT, 1), np.int64) * ((i_idx // 256) % 16),
}


def compile_probe(name, digits):
    W = digits.astype(np.float32).reshape(COUT, CIN, 1, 1)

    BIAS = np.full(COUT, 0.5, np.float32)   # nonzero bias -> emits pico's 128-byte header (0x6480)
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CIN, 1, S))])
    def prog(x):
        return mb.conv(x=x, weight=W, bias=BIAS, strides=[1, 1], pad_type='valid', name='conv')

    m = ct.convert(prog, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS18,
                   compute_precision=ct.precision.FLOAT16)
    m = cto.palettize_weights(m, cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        nbits=4, mode='kmeans', granularity='per_tensor', weight_threshold=256)))
    pkg = name + '.mlpackage'
    subprocess.run(['rm', '-rf', pkg]); m.save(pkg)
    mlc = 'mlc_' + name
    subprocess.run(['rm', '-rf', mlc])
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg, mlc], capture_output=True)
    hx = 'hx_' + name
    subprocess.run(['rm', '-rf', hx])
    subprocess.run([MIL2HWX, '-a', 'h16g', '-i', '%s/%s.mlmodelc/' % (mlc, name), '-o', hx, name],
                   capture_output=True, env=dict(os.environ, ANE_ARCH_ANY='1'))
    h = glob.glob('%s*/model.hwx' % hx)
    if not h:
        return None
    return h[0]


def coeff_banks(path):
    """Locate the 16 banks using the real segment/CoeffSize values from hwx_parsing.

    The coefficient data lives at the start of __KERN_0, and the emitted CoeffSize at this
    mode is 0x6440 = 64-byte header + 25600-byte payload (no scale table -- pico's shipped
    tiles use 0x6480, a 128-byte header that additionally carries 16 fp16 scales).
    Scanning for a "plausible codebook" instead of reading these values landed in __DEBUG
    and produced garbage, so both are parsed rather than guessed.
    """
    d = np.fromfile(path, dtype=np.uint8)
    seg = subprocess.run([HP, '-s', path], capture_output=True, text=True).stdout
    best = None
    lines = seg.splitlines()
    for k, ln in enumerate(lines):
        if 'Segment Name:' in ln and '__KERN_0' in ln:
            for m in lines[k:k + 8]:
                if 'File Off:' in m:
                    best = int(m.split('File Off:')[1].strip(), 16)
                    break
            break
    txt = subprocess.run([HP, '-r', path], capture_output=True, text=True).stdout
    cs = [l for l in txt.splitlines() if 'CoeffSize[0]:' in l]
    stride = int(cs[0].split(':')[1].strip(), 16) if cs else (64 + PAYLOAD_B)
    hdr = stride - PAYLOAD_B
    if best is None:
        return None, None
    banks = []
    for b in range(NBANK):
        p = best + b * stride
        cb = d[p:p + 32].view(np.float16).astype(np.float32)
        raw = d[p + hdr:p + hdr + PAYLOAD_B]
        nib = np.empty(PAYLOAD_NIB, np.uint8)
        nib[0::2] = raw & 0xF
        nib[1::2] = raw >> 4
        banks.append(cb[nib])          # decode through this bank's OWN codebook
    return best, np.array(banks)       # [16, 51200]


print('compiling %d probes at Cin=%d Cout=%d ...' % (len(PROBES), CIN, COUT), flush=True)
vals = {}
for name, dg in PROBES.items():
    h = compile_probe(name, dg)
    if h is None:
        print('  %s: COMPILE FAILED' % name, flush=True); sys.exit(1)
    off, banks = coeff_banks(h)
    if banks is None:
        print('  %s: could not locate banks' % name, flush=True); sys.exit(1)
    vals[name] = banks
    print('  %s ok (banks at 0x%x, %s)' % (name, off, banks.shape), flush=True)

o = (np.rint(vals['o0']) + 16 * np.rint(vals['o1'])).astype(np.int64)
i = (np.rint(vals['i0']) + 16 * np.rint(vals['i1']) + 256 * np.rint(vals['i2'])).astype(np.int64)
print()
print('recovered ranges: o [%d..%d]  i [%d..%d]' % (o.min(), o.max(), i.min(), i.max()))

pairs = set()
for b in range(NBANK):
    pairs.update(zip(o[b].tolist(), i[b].tolist()))
print('distinct (o,i) pairs: %d   expected: %d' % (len(pairs), COUT * CIN))
if len(pairs) == COUT * CIN:
    print('=> PERFECT BIJECTION')

# express the map in closed form: for bank 0, how do o and i advance with slot?
print()
print('bank 0, first 24 slots:  o =', o[0][:24].tolist())
print('bank 0, first 24 slots:  i =', i[0][:24].tolist())
print('bank 0 o range: [%d..%d]   bank 1 o range: [%d..%d]'
      % (o[0].min(), o[0].max(), o[1].min(), o[1].max()))
np.savez(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'posread_L_result.npz'), o=o, i=i)
print('\nsaved results/posread_L_result.npz')
