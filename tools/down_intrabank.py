"""Test a TRANSPOSED intra-bank slot mapping for `down`.

Findings so far: O is stored transposed (#22, independently confirmed by the head
signature). And `down` is misaligned on BOTH axes -- its 3200 axis does not match the
gate/up FFN-neuron axis (-0.044 vs reference +0.222) and its 1024 axis does not match the
residual basis (+0.022 vs +0.700). Both axes failing rules out a pure output permutation
and points at the intra-bank slot->(input, output) mapping itself.

An L-class bank holds 51200 nibbles = 16 outputs x 3200 inputs. The decoder assumes
OUTPUT-fastest:
        o = slot % 16          i = slot // 16
The transposed reading -- the direct analogue of what O turned out to need -- is
INPUT-fastest:
        o = slot // 3200       i = slot % 3200

Both are scored on two oracles that need no forward pass:
  ffn : does the 3200 axis match gate/up's FFN-neuron axis?      (reference +0.222)
  res : does the 1024 axis match the residual basis via gate_in? (reference +0.700)
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

STRIDE, PAY, NOUT = 0x6480, 25600, 16
NIN_BANK = PAY * 2 // NOUT          # 3200


def decode_down_mode(entry, mode):
    W = np.zeros((3200, 1024), np.float32)
    ob = 0
    for off in entry['block_offsets']:
        base = int(off, 16)
        for b in range(16):
            p = base + b * STRIDE
            cb = np.frombuffer(bytes(d[p:p + 32]), dtype=np.float16).astype(np.float32)
            sc = np.frombuffer(bytes(d[p + 64:p + 64 + NOUT * 2]), dtype=np.float16).astype(np.float32)
            r = np.asarray(d[p + 128:p + 128 + PAY])
            nb = np.empty(PAY * 2, np.uint8); nb[0::2] = r & 0xF; nb[1::2] = r >> 4
            slot = np.arange(PAY * 2)
            if mode == 'ofast':                 # current decoder
                o = slot % NOUT; i = slot // NOUT
            else:                               # 'ifast' -- transposed intra-bank
                o = slot // NIN_BANK; i = slot % NIN_BANK
            W[i, ob + b * NOUT + o] = cb[nb] * sc[o]
        ob += 16 * NOUT
    return W


def c(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


NL = 5
print('%-5s %-11s %-11s %-11s %-11s' % ('L', 'ofast ffn', 'ifast ffn', 'ofast res', 'ifast res'))
acc = {k: [] for k in ('of_f', 'if_f', 'of_r', 'if_r')}
ref_f, ref_r = [], []
for L in range(NL):
    ent = by_layer[L]
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)   # [1024,3200]
    U = np.asarray(pw.decode_tensor(ent['up']), dtype=np.float32)
    g_ffn = np.linalg.norm(G, axis=0); u_ffn = np.linalg.norm(U, axis=0)
    g_res = np.linalg.norm(G, axis=1)
    ref_f.append(c(g_ffn, u_ffn)); ref_r.append(c(g_res, np.linalg.norm(U, axis=1)))
    row = []
    for mode in ('ofast', 'ifast'):
        W = decode_down_mode(ent['down'], mode)
        row.append((c(np.linalg.norm(W, axis=1), g_ffn), c(np.linalg.norm(W, axis=0), g_res)))
    acc['of_f'].append(row[0][0]); acc['of_r'].append(row[0][1])
    acc['if_f'].append(row[1][0]); acc['if_r'].append(row[1][1])
    print('%-5d %+.4f     %+.4f     %+.4f     %+.4f'
          % (L, row[0][0], row[1][0], row[0][1], row[1][1]))

print()
print('MEAN  ffn-axis : ofast %+.4f | ifast %+.4f   (reference gate~up %+.4f)'
      % (np.mean(acc['of_f']), np.mean(acc['if_f']), np.mean(ref_f)))
print('MEAN  res-axis : ofast %+.4f | ifast %+.4f   (reference gate~up %+.4f)'
      % (np.mean(acc['of_r']), np.mean(acc['if_r']), np.mean(ref_r)))
