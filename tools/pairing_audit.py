"""Audit EVERY weight pairing in pico with the control-validated composition oracle.

The oracle (||A @ B||_F vs random permutation of the contracted axis) was validated on
Qwen3-4B at z = +300 for a correct pairing, and it has already shown that pico's O must be
TRANSPOSED (z +96 transposed vs +2.9 as decoded).

Now check every place two pico tensors share an index, so the defect can be isolated by
elimination rather than assumed. Each row reports the best orientation and its z.

Pairings (contracted index in brackets):
  O    -> gate      [residual]  attention output feeds the FFN of the same layer
  down -> Q(L+1)    [residual]  FFN output feeds the next layer's attention
  down -> gate(L+1) [residual]  FFN output feeds the next layer's FFN
  gate -> down      [ffn]       gate's neurons are consumed by down
  up   -> down      [ffn]       up's neurons are consumed by down
  Q    -> O         [attn head] Q's head space is where O reads from
"""
import sys, json
import numpy as np
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import pico_weights as pw

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

rng = np.random.RandomState(0)
NL = 5
NR = 10


def z_of(A, B):
    """A: [x, k], B: [k, y]; permute the contracted axis k of A."""
    if A.shape[1] != B.shape[0]:
        return None
    s0 = float(np.linalg.norm(A @ B))
    rs = [float(np.linalg.norm(A[:, rng.permutation(A.shape[1])] @ B)) for _ in range(NR)]
    m, sd = float(np.mean(rs)), float(np.std(rs))
    return (s0 - m) / (sd + 1e-9)


def best_orientation(X, Y):
    """try both orientations of X against Y; return (label, z)."""
    out = []
    for lx, A in (('as-decoded', X), ('transposed', X.T)):
        z = z_of(A, Y)
        if z is not None:
            out.append((z, lx))
    if not out:
        return ('n/a', float('nan'))
    z, lx = max(out)
    return (lx, z)


print('decoding ...', flush=True)
W = {}
for L in range(NL + 1):
    ent = by_layer[L]
    for r in ('Q', 'O', 'gate', 'up', 'down'):
        W[(L, r)] = (np.asarray(pw.decode_down(ent[r]), dtype=np.float32) if r == 'down'
                     else np.asarray(pw.decode_tensor(ent[r]), dtype=np.float32))
print('done\n', flush=True)

tests = [
    ('O -> gate      [residual]', lambda L: (W[(L, 'O')], W[(L, 'gate')])),
    ('down -> Q(L+1) [residual]', lambda L: (W[(L, 'down')], W[(L + 1, 'Q')])),
    ('down -> gate(L+1)[residual]', lambda L: (W[(L, 'down')], W[(L + 1, 'gate')])),
    ('gate -> down    [ffn]', lambda L: (W[(L, 'gate')], W[(L, 'down')])),
    ('up   -> down    [ffn]', lambda L: (W[(L, 'up')], W[(L, 'down')])),
    ('Q -> O          [head]', lambda L: (W[(L, 'Q')], W[(L, 'O')])),
]

print('%-30s %-14s %s' % ('pairing', 'best orient', 'mean z over %d layers' % NL))
for name, get in tests:
    zs, labs = [], []
    for L in range(NL):
        X, Y = get(L)
        lab, z = best_orientation(X, Y)
        if np.isfinite(z):
            zs.append(z); labs.append(lab)
    if not zs:
        print('%-30s %-14s n/a' % (name, '-'))
        continue
    lab = max(set(labs), key=labs.count)
    mz = float(np.mean(zs))
    flag = '  <== ALIGNED' if mz > 20 else ('  <== random' if mz < 5 else '')
    print('%-30s %-14s %+.1f%s' % (name, lab, mz, flag))

print()
print('reference: a correct pairing scores z ~ +300 (Qwen3-4B); random scores ~0')
