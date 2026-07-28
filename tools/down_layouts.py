"""Isolate `down`'s OUTPUT ordering.

Two reasons this is worth redoing separately from #21:

 1. structured_layouts2.py scored O and down JOINTLY. O's fault turns out to be an axis
    TRANSPOSE (#22), not a permutation, so averaging the two roles buries whatever signal
    down carries. Score down alone.

 2. down cannot be transposed the way O was: O is square (1024x1024) so its two axes are
    interchangeable, but down is [3200 in, 1024 out] -- the 3200 axis MUST be the input and
    the 1024 (blk x bank x o) composite MUST be the output. Its axis assignment is forced;
    only the ORDER of the 1024-way composite is open.

And the oracle is unusually clean here: colnorm(down)[o] = ||down[:,o]|| is a norm over the
input axis, so it is INVARIANT to any permutation of the 3200 inputs. A low score therefore
indicts the OUTPUT ordering specifically -- input scrambling cannot explain it.

Tests all 48 (blk,bank,o) digit orderings x reversals, per channel (no grouping confound),
plus the bit-reversal / z-order style shuffles that ANE layouts commonly use.
"""
import sys, json, itertools
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
MASK = ~np.eye(1024, dtype=bool)


def cosmat(A):
    n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return (n @ n.T).astype(np.float32)


def oc(A, B):
    a, b = A[MASK], B[MASK]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print('decoding ...', flush=True)
DN, CD, GN, CG, UN = {}, {}, {}, {}, {}
for L in range(NL):
    ent = by_layer[L]
    D = np.asarray(pw.decode_down(ent['down']), dtype=np.float32)     # [3200,1024]
    G = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)
    U = np.asarray(pw.decode_tensor(ent['up']), dtype=np.float32)
    DN[L] = np.linalg.norm(D, axis=0)          # per OUTPUT channel (input-perm invariant)
    CD[L] = cosmat(D.T)                        # output-direction similarity
    GN[L] = np.linalg.norm(G, axis=1)
    CG[L] = cosmat(G)
    UN[L] = np.linalg.norm(U, axis=1)
    print('  layer %d' % L, flush=True)

ref1 = np.mean([corr(GN[L], UN[L]) for L in range(NL)])
ref2 = np.mean([oc(CG[L], cosmat(np.asarray(pw.decode_tensor(by_layer[L]['up']), dtype=np.float32)))
                for L in range(NL)])
rp = np.random.RandomState(0).permutation(1024)
print()
print('REFERENCE (two known-correct readers): 1st %+.4f | 2nd %+.4f' % (ref1, ref2))
print('SHUFFLED control                    : 1st %+.4f | 2nd %+.4f'
      % (np.mean([corr(DN[L][rp], GN[L]) for L in range(NL)]),
         np.mean([oc(CD[L][np.ix_(rp, rp)], CG[L]) for L in range(NL)])))
print()

d = np.arange(1024)
FIELD = {'blk': d // 256, 'bank': (d // 16) % 16, 'o': d % 16}
SIZE = {'blk': 4, 'bank': 16, 'o': 16}
cands = {}
for order in itertools.permutations(['blk', 'bank', 'o']):
    for rev in itertools.product([False, True], repeat=3):
        t = np.zeros(1024, dtype=np.int64); mult = 1
        for name, r in zip(reversed(order), reversed(rev)):
            f = FIELD[name]
            if r:
                f = SIZE[name] - 1 - f
            t = t + f * mult; mult *= SIZE[name]
        cands['%s%s' % ('/'.join(order), ''.join('R' if r else '.' for r in rev))] = np.argsort(t)

# bit-reversal style shuffles (common in ANE / FFT-ish tile orders)
def bitrev(n, bits):
    r = 0
    for k in range(bits):
        r = (r << 1) | ((n >> k) & 1)
    return r
cands['bitrev10'] = np.argsort(np.array([bitrev(x, 10) for x in d]))
cands['bitrev_bank_only'] = np.argsort((d // 256) * 256 + np.array([bitrev((x // 16) % 16, 4) for x in d]) * 16 + d % 16)
cands['bitrev_o_only'] = np.argsort((d // 16) * 16 + np.array([bitrev(x % 16, 4) for x in d]))

res = []
for lbl, perm in cands.items():
    s1 = np.mean([corr(DN[L][perm], GN[L]) for L in range(NL)])
    s2 = np.mean([oc(CD[L][np.ix_(perm, perm)], CG[L]) for L in range(NL)])
    res.append((s1, s2, lbl))
res.sort(reverse=True)
print('%-26s %-12s %s' % ('candidate', '1st-order', '2nd-order'))
for s1, s2, lbl in res[:12]:
    mk = '   <== current decode' if lbl == 'blk/bank/o...' else ''
    print('%-26s %+.4f      %+.4f%s' % (lbl, s1, s2, mk))
cur = [r for r in res if r[2] == 'blk/bank/o...'][0]
print('...\n%-26s %+.4f      %+.4f   <== current decode' % (cur[2], cur[0], cur[1]))
print()
print('best %s: 1st %+.4f (ref %+.4f) | 2nd %+.4f (ref %+.4f)'
      % (res[0][2], res[0][0], ref1, res[0][1], ref2))
