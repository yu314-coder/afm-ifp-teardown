"""Search `down`'s intra-bank slot mapping with the CONTROL-VALIDATED oracle.

pairing_audit.py isolated the defect: with O transposed, every pairing NOT involving `down`
is aligned (O->gate z=+105, Q->O z=+71) and every pairing involving `down` sits at random
(z between -2.6 and +1.4). Both of down's axes fail simultaneously, which is the signature of
a wrong slot -> (input, output) decomposition inside the bank rather than a permutation of
either axis.

An L-class bank holds 51,200 nibbles = 16 outputs x 3,200 inputs. Candidate decompositions,
all of which are plausible ANE coefficient-group layouts:

    slot = (i // G) * (16*G) + o * G + (i % G)

i.e. the engine walks a chunk of G inputs for output 0, then the same chunk for output 1, ...
    G = 1     -> slot = i*16 + o          (the current decoder: output-fastest)
    G = 3200  -> slot = o*3200 + i        (input-fastest, fully transposed)
    G = 2,4,8,16,32,64 -> intermediate coefficient-group widths

Scored on the validated statistic ||A @ B||_F vs random permutation of the contracted axis,
using two INDEPENDENT pairings that both involve down:
    gate -> down     contracting the FFN axis
    down -> gate(L+1) contracting the residual axis
A correct decomposition must lift BOTH, so the worse of the two is reported.
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/fix/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/fix/pico_shapes')
import pico_weights as pw
import picolib

d = picolib._d
M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

STRIDE, PAY, NOUT, NIN = 0x6480, 25600, 16, 3200
rng = np.random.RandomState(0)
NL = 3
NR = 8


def decode_down_G(entry, G):
    """slot = (i//G)*(16*G) + o*G + (i%G)  ->  invert to get (i, o) per slot."""
    slot = np.arange(PAY * 2)
    chunk = 16 * G
    blkid = slot // chunk            # which input-chunk
    rem = slot % chunk
    o = rem // G
    i = blkid * G + (rem % G)
    W = np.zeros((NIN, 1024), np.float32)
    ob = 0
    for off in entry['block_offsets']:
        base = int(off, 16)
        for b in range(16):
            p = base + b * STRIDE
            cb = np.frombuffer(bytes(d[p:p + 32]), dtype=np.float16).astype(np.float32)
            sc = np.frombuffer(bytes(d[p + 64:p + 64 + NOUT * 2]), dtype=np.float16).astype(np.float32)
            r = np.asarray(d[p + 128:p + 128 + PAY])
            nb = np.empty(PAY * 2, np.uint8); nb[0::2] = r & 0xF; nb[1::2] = r >> 4
            W[i, ob + b * NOUT + o] = cb[nb] * sc[o]
        ob += 16 * NOUT
    return W


def z_of(A, B):
    s0 = float(np.linalg.norm(A @ B))
    rs = [float(np.linalg.norm(A[:, rng.permutation(A.shape[1])] @ B)) for _ in range(NR)]
    m, sd = float(np.mean(rs)), float(np.std(rs))
    return (s0 - m) / (sd + 1e-9)


print('loading readers ...', flush=True)
G_, Gn_ = {}, {}
for L in range(NL + 1):
    G_[L] = np.asarray(pw.decode_tensor(by_layer[L]['gate']), dtype=np.float32)   # [res, ff]
print('done\n', flush=True)

print('%-8s %-14s %-16s %s' % ('G', 'gate->down[ffn]', 'down->gate(L+1)[res]', 'worse'))
best = None
for G in (1, 2, 4, 8, 16, 32, 64, 3200):
    z1s, z2s = [], []
    for L in range(NL):
        D = decode_down_G(by_layer[L]['down'], G)          # [3200 ff, 1024 res]
        z1s.append(z_of(G_[L], D))                          # contract ff
        z2s.append(z_of(D, G_[L + 1]))                      # contract res
    z1, z2 = float(np.mean(z1s)), float(np.mean(z2s))
    w = min(z1, z2)
    tag = ''
    if G == 1:
        tag = '   <== current decoder'
    if best is None or w > best[0]:
        best = (w, G)
    print('%-8s %+-14.1f %+-16.1f %+.1f%s' % (G, z1, z2, w, tag))

print()
print('reference: a correct pairing scores z ~ +300 (Qwen3-4B control); random ~0')
print('best G = %s (worse-of-two z %+.1f)' % (best[1], best[0]))
