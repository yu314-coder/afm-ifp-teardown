"""Does SECOND-ORDER residual structure transfer between roles? (premise test)

#20 showed a per-channel scalar (the norm profile) localises the O/down defect but
carries too little information to identify the permutation. The natural upgrade is a
second-order statistic: for residual positions p,q, how similar are the weight
DIRECTIONS attached to them. That is a 64x64 matrix instead of 64 scalars.

Define, at group granularity (G groups of `gs` consecutive residual channels):
    reader role r  ->  C_r[g,h] = cos-similarity between the mean INPUT direction of
                       group g and of group h   (rows of Q/gate/up, indexed by residual)
    writer role w  ->  C_w[g,h] = same over mean OUTPUT directions (columns of O/down)

PREMISE (must hold before any solve is attempted): if a shared residual geometry drives
this, then two INDEPENDENT known-correct readers must produce entry-wise similar C.
    corr(C_gate, C_up), corr(C_gate, C_Q), corr(C_gate, C_gate(next layer))
Compare against a shuffled control.

If the premise FAILS the QAP route is dead on arrival and is not worth running -- report
that instead of fitting something meaningless.
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

NL = 5
rng = np.random.RandomState(0)


def cosmat(V):
    """V: [G, d] group directions -> GxG cosine similarity."""
    n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def group_dirs(W, axis, gs):
    """axis=0 -> group the INPUT axis (rows, readers). axis=1 -> group OUTPUT axis (cols, writers)."""
    A = W if axis == 0 else W.T          # [residual, other]
    G = A.shape[0] // gs
    return A[:G * gs].reshape(G, gs, A.shape[1]).mean(axis=1)


def offdiag_corr(A, B):
    m = ~np.eye(A.shape[0], dtype=bool)
    a, b = A[m], B[m]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print('decoding ...', flush=True)
W = {}
for L in range(NL + 1):
    ent = by_layer[L]
    for role in ['Q', 'O', 'gate', 'up', 'down']:
        W[(L, role)] = (np.asarray(pw.decode_down(ent[role]), dtype=np.float32) if role == 'down'
                        else np.asarray(pw.decode_tensor(ent[role]), dtype=np.float32))
print('done\n', flush=True)

for gs in (16, 8):
    G = 1024 // gs
    print('=== group size %d (%d groups) ===' % (gs, G))
    print('%-6s %-14s %-14s %-16s %-14s %s'
          % ('layer', 'gate~up', 'gate~Q', 'gate~gate(L+1)', 'shuffled', 'O_out~gate  down_out~gate'))
    r_gu, r_gq, r_gg, r_sh, r_og, r_dg = [], [], [], [], [], []
    for L in range(NL):
        Cg = cosmat(group_dirs(W[(L, 'gate')], 0, gs))
        Cu = cosmat(group_dirs(W[(L, 'up')], 0, gs))
        Cq = cosmat(group_dirs(W[(L, 'Q')], 0, gs))
        Cg2 = cosmat(group_dirs(W[(L + 1, 'gate')], 0, gs))
        Co = cosmat(group_dirs(W[(L, 'O')], 1, gs))       # writer: OUTPUT axis
        Cd = cosmat(group_dirs(W[(L, 'down')], 1, gs))    # writer: OUTPUT axis
        p = rng.permutation(G)
        a, b, c = offdiag_corr(Cg, Cu), offdiag_corr(Cg, Cq), offdiag_corr(Cg, Cg2)
        d = offdiag_corr(Cg, Cu[np.ix_(p, p)])
        e = offdiag_corr(Cg, Co)
        f = offdiag_corr(Cg, Cd)
        r_gu.append(a); r_gq.append(b); r_gg.append(c); r_sh.append(d); r_og.append(e); r_dg.append(f)
        print('%-6d %+.4f        %+.4f        %+.4f          %+.4f        %+.4f      %+.4f'
              % (L, a, b, c, d, e, f))
    print('  MEAN gate~up %+.4f | gate~Q %+.4f | gate~gate(L+1) %+.4f | shuffled %+.4f'
          % (np.mean(r_gu), np.mean(r_gq), np.mean(r_gg), np.mean(r_sh)))
    print('  MEAN O_out~gate %+.4f | down_out~gate %+.4f' % (np.mean(r_og), np.mean(r_dg)))
    strong = np.mean(r_gu) > 0.25 and np.mean(r_gu) > abs(np.mean(r_sh)) + 0.15
    print('  PREMISE %s\n' % ('HOLDS -> QAP worth running' if strong else
                              'FAILS -> second-order structure does not transfer; QAP is dead on arrival'))
