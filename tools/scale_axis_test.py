"""Which axis is the OUTPUT? Settle it with the per-bank SCALES.

Tension to resolve:
  * #22 found O's decoded axis0 carries residual structure and axis1 carries the
    attention-head signature -> "O is stored transposed".
  * But the hwx descriptor says a bank holds 16 OUTPUT channels x all inputs, and each
    bank header stores exactly 16 fp16 scales. A palettization scale is per-OUTPUT-channel
    by construction. That makes the (blk, bank, o) composite the OUTPUT axis -- the opposite
    of #22.

The scales settle it, and they are evidence not used anywhere so far. Collect the 16 scales
from each of the 64 banks (4 blocks x 16) in composite order -> a length-1024 vector indexed
by the composite axis. Then:

  * if the composite axis is the RESIDUAL output, that scale profile must correlate with the
    residual basis (gate's input-norm profile), because a channel's scale sets its magnitude
  * if it does not, the composite is not the residual axis

Run for O and for down. This is a direct read of the storage format rather than an inference
from weight statistics, so it outranks both #22 and #23 where they disagree.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import pico_weights as pw
import picolib

d = picolib._d
M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

GEOM = {'N': (0x2080, 8192, 16), 's': (0x1080, 4096, 8), 'L': (0x6480, 25600, 16)}


def scales_of(entry):
    """per-bank fp16 scales concatenated in composite (blk, bank, o) order."""
    out = []
    for off, cls in zip(entry['block_offsets'], entry['block_classes']):
        stride, pay, nout = GEOM[cls]
        base = int(off, 16)
        for b in range(16):
            p = base + b * stride
            sc = np.frombuffer(bytes(d[p + 64:p + 64 + nout * 2]), dtype=np.float16).astype(np.float32)
            out.append(sc)
    return np.concatenate(out)


def c(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


NL = 8
print('%-5s %-14s %-14s %-14s %-14s' % ('L', 'O scale~gate', 'down scale~gate', 'O wnorm~gate', 'ref gate~up'))
acc = {k: [] for k in ('os', 'ds', 'ow', 'ref')}
for L in range(NL):
    ent = by_layer[L]
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)
    U = np.asarray(pw.decode_tensor(ent['up']), dtype=np.float32)
    g_res = np.linalg.norm(G, axis=1)
    u_res = np.linalg.norm(U, axis=1)
    O = np.asarray(pw.decode_tensor(ent['O']), dtype=np.float32)

    so = scales_of(ent['O'])            # 1024, indexed by composite
    sd = scales_of(ent['down'])         # 1024, indexed by composite
    r = [c(so, g_res), c(sd, g_res), c(np.linalg.norm(O, axis=0), g_res), c(g_res, u_res)]
    for k, val in zip(('os', 'ds', 'ow', 'ref'), r):
        acc[k].append(val)
    print('%-5d %+.4f        %+.4f        %+.4f        %+.4f' % (L, *r))

print()
print('MEAN  O scales~residual   %+.4f' % np.mean(acc['os']))
print('MEAN  down scales~residual %+.4f' % np.mean(acc['ds']))
print('MEAN  O colnorm~residual  %+.4f  (weight-derived, for comparison)' % np.mean(acc['ow']))
print('MEAN  reference gate~up   %+.4f' % np.mean(acc['ref']))
print()
print('Also: do O and down scales agree with EACH OTHER? Both write the same residual basis.')
ag = []
for L in range(NL):
    ent = by_layer[L]
    ag.append(c(scales_of(ent['O']), scales_of(ent['down'])))
print('MEAN  O scales ~ down scales %+.4f' % np.mean(ag))
