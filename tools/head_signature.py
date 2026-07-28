"""Settle O's axis orientation using the ATTENTION HEAD signature (oracle-independent).

#22 found O's rows correlate with the residual basis far better than its columns, which
would mean O is stored with its axes swapped relative to the OutTrans=0 roles. But that
conclusion leans on the residual-basis oracle. This test is independent of it.

Idea: one of O's axes is the attention head-concat (16 heads x 64 = 1024). Head structure
is detectable: dimensions inside the same 64-wide head block are more alike than dimensions
in different heads, so the cosine matrix shows 64-periodic block structure. The other axis is
the residual stream, which has no such 64-periodicity.

Calibration comes from tensors whose orientation is NOT in doubt:
   Q  [1024 residual-in, 1024 head-out]  -> axis1 has head structure, axis0 does not
   gate [1024 residual-in, 3200 ffn-out] -> axis0 is residual (shape is unambiguous)

Statistic: mean cosine WITHIN a 64-block minus mean cosine ACROSS blocks. A positive value
means "this axis is organised in 64-wide head groups".
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/fix/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/fix/pico_shapes')
import pico_weights as pw

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

NL = 6
HD = 64


def cosmat(A):
    n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return (n @ n.T).astype(np.float32)


def head_score(C, hd=HD):
    """mean cos WITHIN a hd-block minus mean cos ACROSS blocks (diagonal excluded)."""
    n = C.shape[0]
    idx = np.arange(n)
    same = (idx[:, None] // hd) == (idx[None, :] // hd)
    off = ~np.eye(n, dtype=bool)
    w = C[same & off].mean()
    a = C[~same].mean()
    return float(w - a)


print('decoding ...', flush=True)
rows = []
for L in range(NL):
    ent = by_layer[L]
    Q = np.asarray(pw.decode_tensor(ent['Q']), dtype=np.float32)      # [res-in, head-out]
    O = np.asarray(pw.decode_tensor(ent['O']), dtype=np.float32)
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)   # [res-in, ffn-out]
    r = {
        'Q axis0 (residual-in, expect ~0)': head_score(cosmat(Q)),
        'Q axis1 (head-out,   expect >0)': head_score(cosmat(Q.T)),
        'gate axis0 (residual, expect ~0)': head_score(cosmat(G)),
        'O axis0': head_score(cosmat(O)),
        'O axis1': head_score(cosmat(O.T)),
    }
    rows.append(r)
    print('  layer %d' % L, flush=True)

print()
keys = list(rows[0].keys())
print('%-36s %s' % ('axis', 'mean 64-block head score'))
for k in keys:
    v = np.mean([r[k] for r in rows])
    print('%-36s %+.5f' % (k, v))

qa0 = np.mean([r['Q axis0 (residual-in, expect ~0)'] for r in rows])
qa1 = np.mean([r['Q axis1 (head-out,   expect >0)'] for r in rows])
oa0 = np.mean([r['O axis0'] for r in rows])
oa1 = np.mean([r['O axis1'] for r in rows])
print()
print('calibration: head-axis %+.5f vs residual-axis %+.5f' % (qa1, qa0))
print()
if abs(oa0 - qa1) < abs(oa0 - qa0) and abs(oa1 - qa0) < abs(oa1 - qa1):
    print('=> O axis0 looks like the HEAD axis and axis1 like the RESIDUAL axis')
    print('   i.e. O is [head-in, residual-out]: the CURRENT decode orientation is correct,')
    print('   and the #22 transpose signal must have another explanation.')
elif abs(oa1 - qa1) < abs(oa1 - qa0) and abs(oa0 - qa0) < abs(oa0 - qa1):
    print('=> O axis1 looks like the HEAD axis and axis0 like the RESIDUAL axis')
    print('   i.e. O IS stored transposed, independently confirming #22.')
else:
    print('=> inconclusive: neither axis matches the calibrated head signature cleanly.')
